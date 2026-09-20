"""Phone-photo bingo sheet reader using the whole-cell CNN.

Pipeline:
  EXIF-correct decode -> exposure-normalize (color-preserving) ->
  teal-band geometry (sheet_to_cells_teal) -> per-cell 75-way logits
  from cell_classifier_phone.pth -> column-constrained decoding.

Column constraints (true for every bingo card):
  B:1-15, I:16-30, N:31-45, G:46-60, O:61-75, with 5 UNIQUE numbers
  per column.  Decoding maximizes the sum of log-softmax scores subject
  to uniqueness via linear assignment in each column.
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms

import photo_sheet_cells
import learning_audit

COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]
FREE = (2, 2)
_SCAN_WIDTH = 1257
# Sheet-level Laplacian variance below this means the photo is out of
# focus (worst training sheet scores 117; every sharp sheet is >= 700), so
# the constrained decode reads mushy cells and commits wrong numbers.  Flag
# those reads for review instead of trusting them.
BLUR_REVIEW_THRESHOLD = 500.0
# A card whose cells decode below these per-cell mean / min probabilities had
# geometry sliced into the wrong rows/columns (pitched close-ups like scan
# 1789934659) and should be flagged for manual review rather than recorded
# silently.  Healthy upright sheets decode well above these (>= ~0.9 mean).
REVIEW_MEAN_CONF = 0.55
REVIEW_MIN_CONF = 0.08

# ImageFolder sorted classes lexicographically ("1","10","11",...), so the
# model's output index does NOT equal the number.  Map between them.
_CLASS_STR = sorted(str(n) for n in range(1, 76))
IDX_TO_NUM = [int(c) for c in _CLASS_STR]
NUM_TO_IDX = {int(c): i for i, c in enumerate(_CLASS_STR)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cell_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def _sheet_blur(gray):
    """Out-of-focus score: Laplacian variance of the normalized sheet."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _teal_hue_stats(bgr):
    """Quick hue/sat/val summary to diagnose why bands weren't found."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    v = hsv[:, :, 2].ravel()
    h[h > 179] = 0
    return (float(np.median(h)),
            [float(np.percentile(h, 20)), float(np.percentile(h, 80))],
            float(np.median(s)), float(np.median(v)))


def _sheet_quad(img):
    """Locate the sheet's outer quadrilateral, or None if it can't be
    trusted (no big bright region, or corners touching the frame edge —
    meaning the page is cut off and warp would be wrong)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0),
                        cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    conts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return None
    big = max(conts, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.10 * H * W:
        return None
    peri = cv2.arcLength(big, True)
    poly = cv2.approxPolyDP(big, 0.02 * peri, True)
    if len(poly) < 4:
        hull = cv2.convexHull(poly)
        poly = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(poly) < 4:
        return None
    pts = poly.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl, tr, br, bl = pts[np.argmin(s)], pts[np.argmax(d)], \
        pts[np.argmax(s)], pts[np.argmin(d)]
    if (tl[0] < 8 or tl[1] < 4 or br[0] > W - 8 or br[1] > H - 4
            or tr[0] > W - 8 or tr[1] < 4 or bl[0] < 8 or bl[1] > H - 4):
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_sheet(img):
    """Perspective-correct the whole sheet to the scan's proportions
    (1257:3475) when it is noticeably trapezoidal (camera pitched).  Near-
    frontal photos are untouched.  Only active with CNN_WARP=1: the model
    was trained on the un-warped geometry, so warping only helps severely
    pitched real-world shots and must be opted into."""
    if os.environ.get("CNN_WARP") != "1":
        return img, False
    q = _sheet_quad(img)
    if q is None:
        return img, False
    tl, tr, br, bl = q
    wtop = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    wbot = np.hypot(br[0] - bl[0], br[1] - bl[1])
    hl = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    hr = np.hypot(br[0] - tr[0], br[1] - tr[1])
    dev = max(abs(wtop - wbot), abs(hl - hr)) / max(wtop, hl)
    if dev < 0.04:
        return img, False
    tw = int(round(max(wtop, wbot)))
    th = int(round(tw * 2.765))
    dst = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (tw, th)), True


def _normalize(img):
    """Downscale to scan scale; stretch luminance so paper is bright while
    preserving hue (teal band detection needs color).  Apply CLAHE to
    enhance local contrast for low-light robustness."""
    H, W = img.shape[:2]
    scale = _SCAN_WIDTH / W
    r = img
    if abs(scale - 1.0) > 0.02:
        r = cv2.resize(img, (_SCAN_WIDTH, int(round(H * scale))),
                       interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    # Apply CLAHE to enhance local contrast, especially helpful in low light
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    y = y.astype(np.float32)
    lo, hi = np.percentile(y, (2, 98))
    y2 = np.clip((y - lo) * (255.0 / max(1.0, hi - lo)), 0, 255)
    yuv[:, :, 0] = y2.astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


class _ConvStack(nn.Module):
    """Feature extractor identical to predict_digit.DigitClassifier."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )

    def forward(self, x):
        return self.features(x)


class CellClassifier(_ConvStack):
    def __init__(self, ncls=76):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, ncls),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path=None):
    """Load the whole-cell phone reader (76 classes: index 0 unused)."""
    path = Path(path or Path(__file__).parent / "cell_classifier_phone.pth")
    model = CellClassifier(76).to(DEVICE)
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_photo_bytes(data):
    """EXIF-correct decode of raw JPEG bytes -> BGR."""
    im = ImageOps.exif_transpose(
        Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def cells_from(sheet_bgr, trace=None):
    """Full-sheet BGR (normalized by caller) -> {cid: {(r, c): cell}} and
    {cid: {(r, c): (xA,yA,xB,yB)}}."""
    items = photo_sheet_cells.sheet_to_cells_teal(
        sheet_bgr, return_geometry=True, trace=trace)
    by = {}
    boxes = {}
    for cid, r, c, cell, box in items:
        by.setdefault(cid, {})[(r, c)] = cell
        boxes.setdefault(cid, {})[(r, c)] = box
    return by, boxes


def cells_from_forced(sheet_bgr, band_tops, trace=None):
    """Same as cells_from but pins the card layout to user-tapped band tops
    (assisted scan for washed-out sheets)."""
    items = photo_sheet_cells.sheet_to_cells_forced(
        sheet_bgr, band_tops, return_geometry=True, trace=trace)
    by = {}
    boxes = {}
    for cid, r, c, cell, box in items:
        by.setdefault(cid, {})[(r, c)] = cell
        boxes.setdefault(cid, {})[(r, c)] = box
    return by, boxes


def _last_card_mean_conf(by_card, cid, model):
    lgrid = {k: cell_logits(model, v) for k, v in by_card[cid].items()}
    _, confs = decode_card(lgrid)
    return float(np.mean([confs[r][c] for r in range(5) for c in range(5)
                          if (r, c) != FREE]))


def _crop_card_cells(sheet_bgr, cb, rows):
    """Re-crop one card's 25 cells directly from the warped sheet: the
    geometry's column lattice (cb) is independent of the row lattice, so a
    candidate row run can be evaluated without recomputing any geometry.
    Cell boxes are returned alongside so an adopted lattice also fixes the
    overlay.  cb/rows are 6-element grids; cells are inset the same way the
    geometry pass crops them."""
    inset = photo_sheet_cells.INSET
    cells, boxes = {}, {}
    for r in range(5):
        yA, yB = rows[r], rows[r + 1]
        if yB - yA <= 2 * inset:
            continue
        for c in range(5):
            xA, xB = cb[c], cb[c + 1]
            if xB - xA <= 2 * inset:
                continue
            cell = sheet_bgr[yA + inset:yB - inset, xA + inset:xB - inset]
            if cell.size == 0:
                continue
            cells[(r, c)] = cell
            boxes[(r, c)] = (xA, yA, xB, yB)
    return cells, boxes


def _adopt_contrast_last_card(model, sheet_bgr, by_card, bboxes, trace):
    """Last-card rows decision (assisted AND auto scans).  The geometry pass
    also derives the grid from a row-wise CONTRAST lattice (see
    _last_card_contrast_lattice), which recovers the true bottom-card pitch
    and first line when perspective makes the pitch grow well past the upper
    cards' median and the faint line 0 hides the printed-grid start.  Whether
    to keep it depends on the model: re-cut card 3 under each distinct
    contrast candidate and adopt the best when its decoded cell confidence
    beats the fallback's.  Re-cuts use the already-computed column lattice
    (same columns, only the rows move), which keeps this cheap -- a full
    per-candidate geometry pass made every auto scan ~5x slower."""
    cards_t = (trace or {}).get("cards") or []
    if not cards_t or 3 not in by_card:
        return
    c3 = cards_t[-1]
    cands = c3.get("rows_contrast") or []
    cb = c3.get("cb_used")
    if not cands or not cb:
        return
    used = c3.get("rows")
    mc_used = _last_card_mean_conf(by_card, 3, model)
    # The contrast sweep emits one variant per line-0 offset (a handful of
    # ~3px steps); keep only genuinely distinct lattices.
    distinct = []
    for cand in cands:
        if cand == used:
            continue
        if any(max(abs(a - b) for a, b in zip(cand, o)) <= 3
               for o in distinct):
            continue
        distinct.append(cand)
    best = None
    for cand in distinct:
        cells3, _ = _crop_card_cells(sheet_bgr, cb, cand)
        if not cells3:
            continue
        mc = _last_card_mean_conf({3: cells3}, 3, model)
        if best is None or mc > best[0]:
            best = (mc, cand, cells3)
    if best is not None and best[0] > mc_used:
        mc_cc, cc, cells3 = best
        by_card[3] = cells3
        _, b3 = _crop_card_cells(sheet_bgr, cb, cc)
        if b3:
            bboxes[3] = b3
        print(f"[geo] card3 contrast re-pick rows {used} -> {cc} "
              f"(mc {mc_used:.2f} -> {mc_cc:.2f})", flush=True)
    elif best is not None:
        mc_cc, cc, _ = best
        print(f"[geo] card3 contrast candidate {cc} REJECTED "
              f"(fallback mc {mc_used:.2f}, best candidate {mc_cc:.2f})",
              flush=True)


def cell_logits(model, cell_bgr):
    """(75,) float32 logits for one BGR or grayscale cell."""
    if cell_bgr.ndim == 3 and cell_bgr.shape[2] == 3:
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    elif cell_bgr.ndim == 2:
        gray = cell_bgr
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {cell_bgr.shape}")
    pil = Image.fromarray(gray)
    x = cell_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0].cpu().numpy().astype(np.float64)
    return logits


def decode_cell_args(logits, col):
    """Raw argmax number for a cell (1..75), no constraints."""
    idx = int(np.argmax(logits))
    n = IDX_TO_NUM[idx] if idx < len(IDX_TO_NUM) else -1
    lo, hi = COLUMN_RANGES[col]
    valid = lo <= n <= hi
    return n, valid


def decode_card(logits_grid, conf_threshold=0.0):
    """logits_grid[5][5] -> (numbers[5][5], conf[5][5]).

    Per column, solve a linear assignment between the 5 non-free cells and
    the 15 legal numbers so each number is used at most once, maximizing
    total log-probability.  Empty dict placeholders decode to None.
    """
    from scipy.optimize import linear_sum_assignment

    numbers = [[None] * 5 for _ in range(5)]
    confs = [[0.0] * 5 for _ in range(5)]
    for c in range(5):
        lo, hi = COLUMN_RANGES[c]
        cand = list(range(lo, hi + 1))
        rows = [r for r in range(5) if (r, c) != FREE and
                logits_grid.get((r, c)) is not None]
        if not rows:
            continue
        logit_for = {}
        for r in rows:
            lg = logits_grid[(r, c)]
            logit_for[r] = np.array([lg[NUM_TO_IDX[n]] for n in cand])
        cost = np.stack([logit_for[r] for r in rows])
        r_idx, n_idx = linear_sum_assignment(-cost)
        for ri, ni in zip(r_idx, n_idx):
            r = rows[ri]
            numbers[r][c] = cand[ni]
            z = logit_for[r] - logit_for[r].max()
            probs = np.exp(z)
            probs /= probs.sum()
            confs[r][c] = float(probs[ni])
    numbers[2][2] = None
    confs[2][2] = 0.0
    return numbers, confs


def card_result(numbers, confs):
    """Shape into the server /scan-card per-cell dict format."""
    grid = []
    for r in range(5):
        row = []
        for c in range(5):
            if (r, c) == FREE:
                row.append({"value": None, "is_free_space": True,
                            "valid": True, "raw": "FREE",
                            "confidence": 100.0})
            else:
                v = numbers[r][c]
                row.append({"value": v, "is_free_space": False,
                            "valid": v is not None,
                            "raw": str(v) if v is not None else None,
                            "confidence": round(100 * confs[r][c], 1)})
        grid.append(row)
    needs_review = any(cell["value"] is None
                       for row in grid for cell in row
                       if not cell["is_free_space"])
    return {"grid": grid, "needs_review": needs_review}


def _recover_lattice(boxes_per_cell):
    """Reconstruct the uniform 6-boundary (rows, cols) lattice a card's
    cell boxes were cut from (boxes_per_cell keys are (r, c))."""
    rows, cols = None, None
    for (r, c), (x0, y0, x1, y1) in boxes_per_cell.items():
        if c == 0:
            if rows is None:
                rows = [None] * 6
            rows[r] = y0
            rows[r + 1] = y1
        if r == 0:
            if cols is None:
                cols = [None] * 6
            cols[c] = x0
            cols[c + 1] = x1
    if (rows is not None and None in rows) or (cols is not None and None in cols):
        return None
    return rows, cols


def _card_decode_batched(model, sheet_bgr, rows, cols, rows_keep=None):
    """Decode one 5x5 card with a SINGLE batched model forward pass.

    rows_keep: iterate rows of rows rows to score (e.g. (0, 4) for a cheap
    pitch/start probe -- the extremes carry the most lattice information).
    Returns (mean_conf, numbers, confs) exactly like the per-cell path, or
    None when any probed cell is too thin to crop (invalid geometry)."""
    if len(rows) != 6 or len(cols) != 6:
        return None
    keep = tuple(range(5)) if rows_keep is None else tuple(rows_keep)
    cells = []
    keys = []
    for r in keep:
        for c in range(5):
            if (r, c) == FREE:
                continue
            yA, yB, xA, xB = rows[r], rows[r + 1], cols[c], cols[c + 1]
            if yB - yA <= 2 * photo_sheet_cells.INSET or \
                    xB - xA <= 2 * photo_sheet_cells.INSET:
                return None
            cell = sheet_bgr[yA + photo_sheet_cells.INSET:yB - photo_sheet_cells.INSET,
                             xA + photo_sheet_cells.INSET:xB - photo_sheet_cells.INSET]
            if cell.size == 0 or cell.shape[0] < 2 or cell.shape[1] < 2:
                return None
            if cell.ndim == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            cells.append(cell_transform(Image.fromarray(cell)))
            keys.append((r, c))
    if not cells:
        return None
    x = torch.stack(cells).to(DEVICE)
    with torch.no_grad():
        logits = model(x).cpu().numpy().astype(np.float64)
    lgrid = {k: logits[j] for j, k in enumerate(keys)}
    numbers, confs = decode_card(lgrid)
    mean_conf = np.mean([confs[r][c] for r in keep for c in range(5)
                         if (r, c) != FREE])
    return float(mean_conf), numbers, confs


def _refit_card_geometry(model, sheet_bgr, card_id, rows, cols,
                         sibling_rows, sibling_cols, base_mean_conf):
    """Recover a low-confidence card's true lattice on a pitched photo.

    A pitched close-up (e.g. scan 1789934659) makes the header-homography
    and dip-snap paths drift: card 1's rebuilt rows came out at an impossible
    107px pitch slicing into card 2's header.  The card's own band (card top
    .. next card's band top) bounds a ~band-pitch grid, so search a small
    lattice family around it and keep the best-decoding one.  Sibling cards
    on the same sheet share the printed grid, so their proven columns are
    tried first.  Only fired on low-confidence NON-last cards; healthy sheets
    never call this (their mean conf is well above the review bar).

    Returns (rows, cols) when a clearly-better lattice is found, else None."""
    if sibling_rows is None:
        return None
    band_top = rows[0]
    band_bottom = sibling_rows[0] - 2
    band_p = float(max(40.0, (band_bottom - band_top) / 5.0))

    # Candidate columns: our own first, then each sibling's (same printed
    # grid, stacked cards), deduplicated.
    col_cands = []
    for cc in [cols, sibling_cols]:
        if cc is not None and cc not in col_cands:
            col_cands.append(list(cc))

    best_cols = cols
    best_cols_conf = base_mean_conf
    for cc in col_cands:
        res = _card_decode_batched(model, sheet_bgr, rows, cc,
                                   rows_keep=(0, 4))
        if res is None:
            continue
        conf, _, _ = res
        if conf > best_cols_conf:
            best_cols_conf = conf
            best_cols = cc

    # Phase 1: cheap probe of the whole band-anchored lattice family, scored
    # on the top and bottom rows only (the extremes carry the pitch/start
    # signal).  Phase 2 confirms the winner with a full-card decode.
    best_probe = (base_mean_conf, None)
    pitches = sorted({round(band_p * f)
                      for f in (0.72, 0.8, 0.85, 0.9, 1.0, 1.1, 1.22)})
    starts = sorted({round(band_top + o * band_p)
                     for o in (-0.35, -0.2, -0.05, 0.1, 0.25)})
    for p in pitches:
        for s in starts:
            r_cand = [s + round(p * k) for k in range(6)]
            if r_cand[-1] > band_bottom + 8:
                continue
            res = _card_decode_batched(model, sheet_bgr, r_cand,
                                       best_cols, rows_keep=(0, 4))
            if res is None:
                continue
            conf, _, _ = res
            if best_probe[1] is None or conf > best_probe[0]:
                best_probe = (conf, r_cand)
    probe_conf, r_probe = best_probe
    if r_probe is None:
        return None
    # Phase 2: full-card decode of the probe winner, its pitch/start
    # neighbors, and the original rows under the better columns.
    best_full = (None, None)
    candidates = [list(rows)]
    p_best = r_probe[1] - r_probe[0]
    s_best = r_probe[0]
    lo_p = max(p_best - 3, int(round(p_best * 0.94)))
    hi_p = min(p_best + 3, int(round(p_best * 1.06)))
    for p in sorted({p_best, lo_p, hi_p}):
        for s in sorted({s_best - 6, s_best, s_best + 6}):
            candidates.append([s + round(p * k) for k in range(6)])
    for r_cand in candidates:
        if r_cand[-1] > band_bottom + 8:
            continue
        res = _card_decode_batched(model, sheet_bgr, r_cand, best_cols)
        if res is None:
            continue
        conf, _, _ = res
        if best_full[0] is None or conf > best_full[0]:
            best_full = (conf, r_cand)
    new_mean, new_rows = best_full
    if new_rows is None:
        return None
    if new_mean >= 0.85 and new_mean > base_mean_conf + 0.10:
        print(f"[refit] card{card_id}: rows {rows} -> {new_rows} "
              f"cols -> {best_cols} mean_conf {base_mean_conf:.3f} -> "
              f"{new_mean:.3f}", flush=True)
        return list(new_rows), list(best_cols)
    return None


def _scale_boxes(boxes, orig_shape, new_shape):
    """Scale box coordinates from original image to normalized image."""
    oh, ow = orig_shape[:2]
    nh, nw = new_shape[:2]
    sx, sy = nw / ow, nh / oh
    scaled = []
    for x0, y0, x1, y1 in boxes:
        scaled.append((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
    return scaled


def read_sheet_bytes(model, data, forced_bands=None):
    """Read a full photo (raw bytes) -> server-shaped /scan-card payload.
    forced_bands: 3 y-rows (user taps) pinning each card's header band."""
    orig = load_photo_bytes(data)
    # Detect card geometry on ORIGINAL image (before normalization)
    # because normalization changes color statistics and breaks header detection.
    from photo_sheet_cells import teal_card_bboxes
    orig_boxes = teal_card_bboxes(orig) if not forced_bands else None
    norm = _normalize(orig)
    if orig_boxes is not None:
        norm_boxes = _scale_boxes(orig_boxes, orig.shape, norm.shape)
    else:
        norm_boxes = None
    # If original detection found fewer than 3 cards, try detecting on
    # normalized image (normalization can reveal faded headers).
    if norm_boxes is not None and len(norm_boxes) < 3:
        norm_detected = teal_card_bboxes(norm)
        # Only upgrade to normalized detection if it finds a full sheet (3 cards)
        # AND the grid detector on the original confirms at least 2 cards.
        # This avoids false positives from enhanced teal on partial sheets.
        if len(norm_detected) == 3:
            from photo_sheet_cells import grid_line_card_bboxes
            grid_boxes, _, truncated = grid_line_card_bboxes(orig)
            if len(grid_boxes) >= 2 and not truncated:
                norm_boxes = norm_detected

    # Last-resort fallback for washed-out gray sheets: when color-based
    # header detection finds fewer than a full sheet, infer card tops from
    # the printed grid lines (header-color-independent).  Only overrides
    # when it finds at least two COMPLETE cards (not truncated).
    # If the fallback detects truncation, return a retake error directly
    # to force assisted scan (don't use bogus color boxes).
    pre_row_bounds = None
    # Assisted (forced) scans trust the user's taps -- skip the auto-only
    # fallback/truncation gates entirely (a washed-out sheet can trip the
    # grid-truncation hedge and spuriously reject a scan that already has 3
    # user-confirmed header bars).
    if not forced_bands and (norm_boxes is None or len(norm_boxes) < 3):
        from photo_sheet_cells import grid_line_card_bboxes
        grid_boxes, grid_row_bounds, truncated = grid_line_card_bboxes(orig)
        if len(grid_boxes) >= 2 and not truncated:
            norm_boxes = _scale_boxes(grid_boxes, orig.shape, norm.shape)
            # Scale row bounds to normalized coordinates
            h_scale = norm.shape[0] / orig.shape[0]
            pre_row_bounds = [[int(y * h_scale) for y in rb]
                              for rb in grid_row_bounds]
        elif truncated:
            # Fallback found cards but they're truncated — return retake
            # error to force assisted scan (don't use bogus color boxes).
            hue = _teal_hue_stats(norm)
            ts = int(time.time())
            dbg = Path("/tmp/cnn_reader_debug")
            dbg.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg / f"in_{ts}.png"), norm)
            overlay = norm.copy()
            cv2.imwrite(str(dbg / f"overlay_{ts}.png"), overlay)
            return {"cards": [], "error": (
                "photo is truncated (missing top rows of lower cards) — "
                "retake with the full sheet in frame, or tap the three "
                "header bars for an assisted scan"), "debug": {
                "dump_in": str(dbg / f"in_{ts}.png"),
                "dump_overlay": str(dbg / f"overlay_{ts}.png"),
                "ts": str(ts),
                "dump_orig": str(dbg / f"orig_{ts}.jpg"),
                "hue_med": round(float(hue[0])),
                "hue_pct20_80": [round(float(x), 1) for x in hue[1]],
                "sat_med": round(float(hue[2])),
"val_med": round(float(hue[3])),
                "in_h": int(norm.shape[0])}}

    # Keep the RAW phone bytes next to the normalized dump so a misread can
    # be reproduced exactly: normalization + warp destroy the original color
    # statistics that the first (teal) detection stage relies on.  Uses the
    # same 1s-precision ts as _read_sheet's in_/overlay_ dumps so the files
    # line up for post-mortems.
    ts = int(time.time())
    dbg = Path("/tmp/cnn_reader_debug")
    dbg.mkdir(parents=True, exist_ok=True)
    (dbg / f"orig_{ts}.jpg").write_bytes(data)

    return _read_sheet(model, norm, forced_bands, pre_boxes=norm_boxes,
                        pre_row_bounds=pre_row_bounds, ts=ts)


def read_sheet_path(model, path):
    """Read a full photo from disk -> server-shaped /scan-card payload."""
    data = Path(path).read_bytes()
    return read_sheet_bytes(model, data)


def warped_sheet_height(data):
    """Height of the normalized+warped sheet for raw photo bytes — the
    coordinate space /scan-assist's band_tops must ultimately land in
    (matches the in_*.png debug dump).  Used to scale client taps that
    arrive in the downscaled served-image pixel space."""
    orig = load_photo_bytes(data)
    return _warp_sheet(_normalize(orig))[0].shape[0]


def _structure_ok(bboxes, band_tops=None):
    """Validate the detected card layout looks like a real sheet.

    Accepts a single card (1 teal band) or a full 3-card sheet whose
    headers are near-equal pitch and width.  Rejects ambiguous layouts
    (extra bands, wildly uneven spacing) so the reader reports a retake
    error instead of emitting garbage grids.

    Pitch is measured on the header bands when available (stable sheet
    feature); a promo/QR plaque between a card's band and its grid shifts
    the grid down, so grid-based pitch would falsely reject an otherwise
    fine sheet.
    """
    n = len(bboxes)
    if n == 0:
        return False, ("no header bands detected - reframe so the sheet "
                       "fills the frame (all 3 cards, flat and top-lit) "
                       "and try again")
    if n == 1:
        return True, ""
    if n > 3:
        return False, (f"detected {n} cards, expected 1 or 3 - the photo "
                       "shows extra teal regions (other sheets in frame?)")
    if band_tops is not None and len(band_tops) == n:
        tops = list(band_tops)
    else:
        tops = [min(v[1] for v in cell_boxes.values())
                for cell_boxes in bboxes.values()]
    widths = [max(v[2] for v in cell_boxes.values())
              - min(v[0] for v in cell_boxes.values())
              for cell_boxes in bboxes.values()]
    pitches = [tops[i] - tops[i - 1] for i in range(1, n)]
    wmax = max(widths)
    pitch_ok = (max(pitches) - min(pitches)) <= 0.28 * max(pitches)
    width_ok = min(widths) >= 0.45 * wmax
    if not (pitch_ok and width_ok):
        return False, (f"card layout is uneven (pitches {pitches}, widths "
                       f"{widths}) - retake flat with the whole sheet in "
                       "frame")
    return True, ""


def _read_sheet(model, sheet_bgr, forced_bands=None, pre_boxes=None,
                pre_row_bounds=None, ts=None):
    """sheet_bgr (normalized full-sheet BGR) -> /scan-card payload.

    forced_bands: when the auto header-color detection fails (washed-out
    gray sheets), the caller supplies 3 tapped rows — one per card's band
    top — and the card geometry is pinned to those instead.

    pre_boxes: pre-detected card header boxes [(x0,y0,x1,y1), ...] in
    sheet_bgr's coordinate space. If provided, these are used instead of
    running header detection on sheet_bgr (which may have degraded colors
    from normalization).

    Always dumps the normalized input + a cell-box overlay to
    /tmp/cnn_reader_debug (with per-card ink/confidence stats) so any
    misread can be diagnosed after the fact.
    """
    sheet_bgr, warped = _warp_sheet(sheet_bgr)
    dbg_h = int(sheet_bgr.shape[0])
    blur = _sheet_blur(cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY))
    blurry = blur < BLUR_REVIEW_THRESHOLD
    trace = {}
    if forced_bands:
        by_card, bboxes = cells_from_forced(sheet_bgr, forced_bands, trace)
        _adopt_contrast_last_card(model, sheet_bgr, by_card, bboxes, trace)
    elif pre_boxes is not None:
        # Use pre-detected boxes (from original image, scaled to normalized)
        from photo_sheet_cells import _sheet_to_cells_with_boxes
        items = _sheet_to_cells_with_boxes(
            sheet_bgr, pre_boxes, return_geometry=True, trace=trace,
            pre_row_bounds=pre_row_bounds, trust_box_x=True)
        by, boxes = {}, {}
        for cid, r, c, cell, box in items:
            by.setdefault(cid, {})[(r, c)] = cell
            boxes.setdefault(cid, {})[(r, c)] = box
        by_card, bboxes = by, boxes
        _adopt_contrast_last_card(model, sheet_bgr, by_card, bboxes, trace)
    else:
        by_card, bboxes = cells_from(sheet_bgr, trace)
    for c in trace.get("cards", []):
        dbg_line = (f"[geo] card{c['card']} rows={c['rows']} "
                    f"cols={c['cb_used']} ext={trace.get('x0')}..{trace.get('x1')}"
                    f" hom={'Y' if c['use_hom'] else 'N'}"
                    f" rewire={'Y' if c['rewired'] else 'N'}")
        if c.get("cb_used") != c.get("cb_eq"):
            dbg_line += f" (eq {c['cb_eq']})"
        print(dbg_line, flush=True)
    if trace and (trace.get("clusters") or trace.get("lattice")):
        print(f"[geo] clusters={trace.get('clusters')} "
              f"lattice={trace.get('lattice')} "
              f"hom_all={trace.get('hom_all')}", flush=True)
    if ts is None:
        ts = int(time.time())
    dbg = Path("/tmp/cnn_reader_debug")
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / f"in_{ts}.png"), sheet_bgr)
    overlay = sheet_bgr.copy()
    for cid, cells in bboxes.items():
        for (r, c), box in cells.items():
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]),
                          (0, 220, 0), 2)
    cv2.imwrite(str(dbg / f"overlay_{ts}.png"), overlay)
    band_tops = [c.get("top") for c in trace.get("cards", [])]
    if not all(isinstance(t, int) for t in band_tops):
        band_tops = None
    ok, msg = _structure_ok(bboxes, band_tops)
    if not ok:
        if forced_bands:
            msg = ("assisted scan couldn't verify a valid 3-card layout - "
                   "tap the top edge of each gray bar more precisely")
        hue = _teal_hue_stats(sheet_bgr)
        return {"cards": [], "error": msg, "debug": {
            "dump_in": str(dbg / f"in_{ts}.png"),
            "dump_overlay": str(dbg / f"overlay_{ts}.png"),
            "ts": str(ts),
            "dump_orig": str(dbg / f"orig_{ts}.jpg"),
            "bands": sorted({v[1] for cell_boxes in bboxes.values()
                             for v in cell_boxes.values()}) if bboxes else [],
            "hue_med": round(float(hue[0])),
            "hue_pct20_80": [round(float(x), 1) for x in hue[1]],
            "sat_med": round(float(hue[2])),
            "val_med": round(float(hue[3])),
            "in_h": dbg_h}}
    if not by_card:
        if forced_bands:
            msg = ("assisted scan found no cells at the tapped rows - tap "
                   "the top edge of each gray bar directly")
        else:
            msg = ("no header bands detected - reframe so the sheet fills "
                   "the frame (all 3 cards, flat and top-lit) and try again")
        hue = _teal_hue_stats(sheet_bgr)
        return {"cards": [], "error": msg, "debug": {
            "dump_in": str(dbg / f"in_{ts}.png"),
            "dump_overlay": str(dbg / f"overlay_{ts}.png"),
            "ts": str(ts),
            "hue_med": round(float(hue[0])),
            "hue_pct20_80": [round(float(x), 1) for x in hue[1]],
            "sat_med": round(float(hue[2])),
            "val_med": round(float(hue[3])),
            "in_h": dbg_h}}
    if forced_bands and len(by_card) != 3:
        # Assisted scans are pinned to 3 user taps — silently emitting 2
        # cards means a tap landed on the wrong bar, so error for a re-tap.
        return {"cards": [], "error": ("assisted scan couldn't verify all "
                "3 cards - tap the top edge of each gray bar precisely and "
                "try again"), "debug": {
            "dump_in": str(dbg / f"in_{ts}.png"),
            "dump_overlay": str(dbg / f"overlay_{ts}.png"),
            "ts": str(ts),
            "dump_orig": str(dbg / f"orig_{ts}.jpg"),
            "bands": sorted({v[1] for cell_boxes in bboxes.values()
                             for v in cell_boxes.values()}) if bboxes else [],
            "in_h": dbg_h}}

    cards = []
    lattices = {}
    for cid in bboxes:
        rec = _recover_lattice(bboxes[cid])
        if rec is not None:
            lattices[cid] = rec
    last_cid = max(by_card) if by_card else None
    refit_rows, refit_cols = {}, {}
    any_refit = False
    for cid in sorted(by_card):
        lgrid = {}
        for (r, c), cell in by_card[cid].items():
            lgrid[(r, c)] = cell_logits(model, cell)
        numbers, confs = decode_card(lgrid)
        result = card_result(numbers, confs)
        if blurry:
            result["needs_review"] = True
        mean_conf = np.mean([confs[r][c] for r in range(5) for c in range(5)
                             if (r, c) != FREE])
        min_conf = min(confs[r][c] for r in range(5) for c in range(5)
                       if (r, c) != FREE)
        if (mean_conf < REVIEW_MEAN_CONF and cid != last_cid and
                cid in lattices and (cid + 1) in lattices):
            # Low-confidence non-last card on (likely) a pitched photo: the
            # geometry may be sliced.  Try the band-anchored lattice family
            # with the sibling card's proven columns (same printed grid).
            fit = _refit_card_geometry(
                model, sheet_bgr, cid,
                lattices[cid][0], lattices[cid][1],
                lattices[cid + 1][0], lattices[cid + 1][1],
                float(mean_conf))
            if fit is not None:
                nrows, ncols = fit
                refit_rows[cid] = list(nrows)
                refit_cols[cid] = list(ncols)
                any_refit = True
                by_card[cid] = {}
                bboxes[cid] = {}
                for r in range(5):
                    for c in range(5):
                        if (r, c) == FREE:
                            continue
                        yA, yB, xA, xB = (nrows[r], nrows[r + 1],
                                          ncols[c], ncols[c + 1])
                        if yB - yA <= 2 * photo_sheet_cells.INSET or \
                                xB - xA <= 2 * photo_sheet_cells.INSET:
                            continue
                        cell = sheet_bgr[yA + photo_sheet_cells.INSET:
                                         yB - photo_sheet_cells.INSET,
                                         xA + photo_sheet_cells.INSET:
                                         xB - photo_sheet_cells.INSET]
                        if cell.size == 0:
                            continue
                        by_card[cid][(r, c)] = cell
                        bboxes[cid][(r, c)] = (xA, yA, xB, yB)
                lgrid = {}
                for (r, c), cell in by_card[cid].items():
                    lgrid[(r, c)] = cell_logits(model, cell)
                numbers, confs = decode_card(lgrid)
                result = card_result(numbers, confs)
                if blurry:
                    result["needs_review"] = True
                mean_conf = np.mean([confs[r][c] for r in range(5)
                                     for c in range(5) if (r, c) != FREE])
                min_conf = np.min([confs[r][c] for r in range(5)
                                   for c in range(5) if (r, c) != FREE])
        if mean_conf < REVIEW_MEAN_CONF or min_conf < REVIEW_MIN_CONF:
            result["needs_review"] = True
        cards.append(result)

    if any_refit:
        # Keep the debug overlay + trace consistent with the refitted
        # geometry the numbers actually came from.
        overlay = sheet_bgr.copy()
        for cid, cells in bboxes.items():
            for (r, c), box in cells.items():
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]),
                              (0, 220, 0), 2)
        cv2.imwrite(str(dbg / f"overlay_{ts}.png"), overlay)
        for c in trace.get("cards", []):
            cid = c.get("card")
            if cid in refit_rows:
                c["rows"] = refit_rows[cid]
                c["cb_used"] = refit_cols[cid]
                c["refit"] = True

    if by_card:
        auto_grids = []
        for card in cards:
            auto_grids.append(
                [[cell["value"] for cell in row] for row in card["grid"]])
        _persist_pending_cells(sheet_bgr, by_card, ts, auto_grids)

    return {"cards": cards, "scan_id": str(ts), "debug": {
        "dump_in": str(dbg / f"in_{ts}.png"),
        "dump_overlay": str(dbg / f"overlay_{ts}.png"),
        "ts": str(ts),
        "dump_orig": str(dbg / f"orig_{ts}.jpg"),
        "partial_sheet": len(by_card) < 3,
        "warped": bool(warped),
        "blur": round(blur, 1),
        "blurry": blurry,
        "geometry": trace,
        "in_h": dbg_h}}


PENDING_DIR = Path("/tmp/phone_learning_pending")


def _persist_pending_cells(sheet_bgr, by_card, scan_id, auto_grids=None):
    """Save the raw cell crops for a successful read so a later user
    confirmation (/cards POST with the same scan_id) can pair each cell
    with its corrected number and grow the training set.  Grayscale
    per-cell JPEGs, same stats as train_phone_cells.build writes.  Blank
    cells are skipped so we never label empty crops.

    auto_grids (list of 5x5 value grids, one per card in cid order) is
    dumped next to the crops as auto.json so the confirmation path can
    diff the user's final grids against what the reader auto-recognized
    and tag cells the user actually corrected."""
    out = PENDING_DIR / str(scan_id)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid in by_card:
        for (r, c), cell in by_card[cid].items():
            if (r, c) == FREE:
                continue
            g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            if (g < 150).mean() < 0.01:
                continue
            cv2.imwrite(str(out / f"c{cid}_r{r}c{c}.jpg"), g)
            n += 1
    if auto_grids is not None:
        learning_audit.dump_auto_json(out / "auto.json", auto_grids)
    return n


if __name__ == "__main__":
    model = load_model()
    for arg in sys.argv[1:]:
        result = read_sheet_path(model, arg)
        result.pop("cards", None)
        print(json.dumps(result, indent=2))