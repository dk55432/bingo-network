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

COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]
FREE = (2, 2)
_SCAN_WIDTH = 1257
# Sheet-level Laplacian variance below this means the photo is out of
# focus (worst training sheet scores 117; every sharp sheet is >= 700), so
# the constrained decode reads mushy cells and commits wrong numbers.  Flag
# those reads for review instead of trusting them.
BLUR_REVIEW_THRESHOLD = 500.0

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
    preserving hue (teal band detection needs color)."""
    H, W = img.shape[:2]
    scale = _SCAN_WIDTH / W
    r = img
    if abs(scale - 1.0) > 0.02:
        r = cv2.resize(img, (_SCAN_WIDTH, int(round(H * scale))),
                       interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32)
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
        if len(norm_detected) > len(norm_boxes):
            norm_boxes = norm_detected
    return _read_sheet(model, norm, forced_bands, pre_boxes=norm_boxes)


def read_sheet_path(model, path):
    """Read a full photo from disk -> server-shaped /scan-card payload."""
    data = Path(path).read_bytes()
    return read_sheet_bytes(model, data)


def _structure_ok(bboxes):
    """Validate the detected card layout looks like a real sheet.

    Accepts a single card (1 teal band) or a full 3-card sheet whose
    headers are near-equal pitch and width.  Rejects ambiguous layouts
    (extra bands, wildly uneven spacing) so the reader reports a retake
    error instead of emitting garbage grids.
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


def _read_sheet(model, sheet_bgr, forced_bands=None, pre_boxes=None):
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
    blur = _sheet_blur(cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY))
    blurry = blur < BLUR_REVIEW_THRESHOLD
    trace = {}
    if forced_bands:
        by_card, bboxes = cells_from_forced(sheet_bgr, forced_bands, trace)
    elif pre_boxes is not None:
        # Use pre-detected boxes (from original image, scaled to normalized)
        from photo_sheet_cells import _sheet_to_cells_with_boxes
        items = _sheet_to_cells_with_boxes(
            sheet_bgr, pre_boxes, return_geometry=True, trace=trace)
        by, boxes = {}, {}
        for cid, r, c, cell, box in items:
            by.setdefault(cid, {})[(r, c)] = cell
            boxes.setdefault(cid, {})[(r, c)] = box
        by_card, bboxes = by, boxes
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
    ok, msg = _structure_ok(bboxes)
    if not ok:
        if forced_bands:
            msg = ("assisted scan couldn't verify a valid 3-card layout - "
                   "tap the top edge of each gray bar more precisely")
        hue = _teal_hue_stats(sheet_bgr)
        return {"cards": [], "error": msg, "debug": {
            "dump_in": str(dbg / f"in_{ts}.png"),
            "dump_overlay": str(dbg / f"overlay_{ts}.png"),
            "ts": str(ts),
            "bands": sorted({v[1] for cell_boxes in bboxes.values()
                             for v in cell_boxes.values()}) if bboxes else [],
            "hue_med": round(float(hue[0])),
            "hue_pct20_80": [round(float(x), 1) for x in hue[1]],
            "sat_med": round(float(hue[2])),
            "val_med": round(float(hue[3]))}}
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
            "val_med": round(float(hue[3]))}}
    if forced_bands and len(by_card) != 3:
        # Assisted scans are pinned to 3 user taps — silently emitting 2
        # cards means a tap landed on the wrong bar, so error for a re-tap.
        return {"cards": [], "error": ("assisted scan couldn't verify all "
                "3 cards - tap the top edge of each gray bar precisely and "
                "try again"), "debug": {
            "dump_in": str(dbg / f"in_{ts}.png"),
            "dump_overlay": str(dbg / f"overlay_{ts}.png"),
            "ts": str(ts),
            "bands": sorted({v[1] for cell_boxes in bboxes.values()
                             for v in cell_boxes.values()}) if bboxes else []}}

    cards = []
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
        if mean_conf < 0.25 or min_conf < 0.08:
            result["needs_review"] = True
        cards.append(result)

    if by_card:
        _persist_pending_cells(sheet_bgr, by_card, ts)

    return {"cards": cards, "scan_id": str(ts), "debug": {
        "dump_in": str(dbg / f"in_{ts}.png"),
        "dump_overlay": str(dbg / f"overlay_{ts}.png"),
        "ts": str(ts),
        "partial_sheet": len(by_card) < 3,
        "warped": bool(warped),
        "blur": round(blur, 1),
        "blurry": blurry,
        "geometry": trace}}


PENDING_DIR = Path("/tmp/phone_learning_pending")


def _persist_pending_cells(sheet_bgr, by_card, scan_id):
    """Save the raw cell crops for a successful read so a later user
    confirmation (/cards POST with the same scan_id) can pair each cell
    with its corrected number and grow the training set.  Grayscale
    per-cell JPEGs, same stats as train_phone_cells.build writes.  Blank
    cells are skipped so we never label empty crops."""
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
    return n


if __name__ == "__main__":
    model = load_model()
    for arg in sys.argv[1:]:
        result = read_sheet_path(model, arg)
        result.pop("cards", None)
        print(json.dumps(result, indent=2))