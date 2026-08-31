"""Phone-photo sheet -> cells, using the server pipeline's Hough geometry.

The scan-tuned sheet_to_cells assumes flatbed-bright paper (fixed <190
threshold); phone photos are darker/noisier, so its thin grid line detector
floods and rows drift. This variant keeps the hue-based teal band detector
for card tops but fits each card's grid with the server pipeline's
Hough-based line detection (pipeline.find_grid_line_positions) + equal-step
boundary snapping, which tolerate lighting gradients and noisy cells.

Caller should pass an already-normalized (brightened) BGR image, e.g. via
evaluate_photo_sheets.normalize.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import _cell_boundaries, find_grid_line_positions  # noqa: E402
from extract_scan_cells import (  # noqa: E402
    INSET,
    _BAND_FAMILIES,
    _header_hue_spec,
    _hue_in,
)


def sheet_to_cells_photo(img, verbose=False, return_geometry=False):
    """Return list of (card_id, row, col, cell) with optional geometry,
    mirroring extract_scan_cells.sheet_to_cells's contract."""
    bands = find_header_bands(img)
    if not bands:
        return []
    H, W = img.shape[:2]
    cells_out = []
    for i, (ts, te) in enumerate(bands):
        top = te + 6
        bottom = bands[i + 1][0] if i + 1 < len(bands) else H
        if bottom - top < 60:
            if verbose:
                print(f"card{i + 1}: region too thin ({bottom - top}px)")
            continue
        region = img[top:bottom, :]
        hy, vx = find_grid_line_positions(region)
        if verbose:
            print(f"card{i + 1}: region ({top}..{bottom}) hlines={hy} vlines={vx}")
        if len(hy) < 4 or len(vx) < 4:
            if verbose:
                print(f"card{i + 1}: not enough lines (h={len(hy)} v={len(vx)})")
            continue
        rb = _cell_boundaries(hy, region.shape[0])
        cb = _cell_boundaries(vx, region.shape[1])
        if verbose:
            print(f"card{i + 1}: row bounds {rb}, col bounds {cb}")
        for r in range(5):
            for c in range(5):
                y0, y1 = top + rb[r], top + rb[r + 1]
                x0, x1 = cb[c], cb[c + 1]
                if y1 - y0 <= 2 * INSET or x1 - x0 <= 2 * INSET:
                    continue
                cell = img[y0 + INSET:y1 - INSET, x0 + INSET:x1 - INSET]
                if cell.size == 0:
                    continue
                if return_geometry:
                    cells_out.append((i + 1, r, c, cell,
                                      (x0, y0, x1, y1)))
                else:
                    cells_out.append((i + 1, r, c, cell))
    return cells_out


def teal_card_bboxes(img):
    """Return [(x0, y0, x1, y1)] per card header band, merged across hue
    splits, sorted by y.  Works for any sheet print color (the header
    color is auto-detected, see _header_hue_spec)."""
    spec = _header_hue_spec(img)
    if spec is None:
        return []
    hl, hh, s_min, v_min, rowfrac = spec
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int)
    s = hsv[:, :, 1].astype(int)
    v = hsv[:, :, 2].astype(int)
    teal = _hue_in(h, hl, hh) & (s > s_min) & (v > v_min)
    rowcnt = teal.mean(axis=1)
    rows = [y for y in range(len(rowcnt)) if rowcnt[y] > rowfrac]
    runs = []
    for y in rows:
        if runs and y - runs[-1][-1] <= 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    runs = [r for r in runs if len(r) >= 15]
    merged = []
    for r in runs:
        if (merged and r[0] - merged[-1][1] < 120
                and cv2.countNonZero(teal[merged[-1][1]:r[0], :]) == 0):
            merged[-1][1] = r[-1]
        else:
            merged.append([r[0], r[-1]])
    boxes = []
    for y0, y1 in merged:
        cols = np.where(teal[y0:y1 + 1, :].any(axis=0))[0]
        if cols.size == 0:
            continue
        boxes.append((int(cols[0]), y0, int(cols[-1]), y1))
    boxes = sorted(boxes, key=lambda b: b[1])
    family_specs = {(hl, hh, s_min, v_min, rowfrac)
                    for (hl, hh), s_min, v_min, rowfrac in _BAND_FAMILIES}
    if spec not in family_specs:
        # Last-resort (adaptive dominant-hue) masks cover the whole frame on
        # washed photos, so the boxes poke off the sheet onto the table.
        # Clamp them to the paper's bright column extent; real family masks
        # are the printed color itself and stay untouched.
        ext = _frame_bright_extent(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ext is not None:
            x_left, x_right = ext
            boxes = [(max(x0, x_left), y0, min(x1, x_right), y1)
                     for x0, y0, x1, y1 in boxes]
    return boxes


def _snap(bounds, dark, lo, hi, radius):
    """Move each boundary to the darkest row/col within radius of it,
    clamped to [lo + 2, hi - 2]. dark: array of darkness per line."""
    out = list(bounds)
    for i in range(1, len(out) - 1):
        a, b = max(lo + 2, out[i] - radius), min(hi - 2, out[i] + radius)
        if b < a:
            continue
        out[i] = a + int(np.argmax(dark[a:b + 1]))
    return out


def _card_y_extent(gray, top, hint_bottom):
    """Clip the card's bottom to the last bright (paper) row in the region,
    so the last card doesn't drag cells into the dark background. Row
    MEDIAN resists bright specks in the dark backdrop that defeat the mean."""
    reg = gray[top:hint_bottom, :].astype(np.float32)
    if reg.size == 0:
        return hint_bottom
    rowmed = np.median(reg, axis=1)
    hi = np.percentile(rowmed, 90)
    thr = max(40, hi - 35)
    below = np.where(rowmed > thr)[0]
    if below.size == 0:
        return hint_bottom
    return min(hint_bottom, top + int(below[-1]) + 2)


def _grid_dip(gray, x0, x1, top, bottom, strip_w=12):
    """Per-row brightness-dip score in a narrow vertical strip centred on
    the card's column extent.  Printed grid lines are thin dark horizontal
    rows that are much darker than their neighbours; a full-width mean
    dilutes them into noise, but a narrow strip preserves the signal."""
    H = gray.shape[0]
    xm = (x0 + x1) // 2
    xL = max(0, xm - strip_w // 2)
    xR = min(gray.shape[1], xm + strip_w // 2)
    strip = gray[top:bottom, xL:xR].mean(axis=1).astype(np.float64)
    n = len(strip)
    score = np.zeros(n)
    for j in range(5, n - 5):
        neighbours = np.concatenate([strip[j - 5:j - 1], strip[j + 2:j + 6]])
        score[j] = neighbours.mean() - strip[j]
    full = np.zeros(H)
    full[top:top + n] = score
    return full


def _sheet_to_cells_with_boxes(
        img, boxes, verbose=False, return_geometry=False):
    """Shared per-card cell extraction for a list of (x0, y0, x1, y1) band
    boxes (card top, fallback estimate) plus bright-paper extent per card
    for the column bounds.  Equal-division rows/cols are snapped toward
    the printed grid rules detected as thin dark horizontal lines via a
    brightness-dip signal in a narrow strip (full-width means dilute the
    thin lines into noise)."""
    if not boxes:
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells_out = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        top = y1 + 2
        bottom = boxes[i + 1][1] if i + 1 < len(boxes) else H
        if i + 1 == len(boxes):
            bottom = _card_y_extent(gray, top, bottom)
        grid_sig = _grid_dip(gray, x0, x1, top, bottom)
        rb = [top + (bottom - top) * k // 5 for k in range(6)]
        rb = _snap(rb, grid_sig, top, bottom, radius=40)
        cb = [x0 + (x1 - x0) * k // 5 for k in range(6)]
        if verbose:
            print(f"card{i + 1}: band ({x0},{y0})-({x1},{y1}) "
                  f"card top={top} bottom={bottom} "
                  f"rows={rb} cols={cb}")
        for r in range(5):
            for c in range(5):
                yA, yB, xA, xB = rb[r], rb[r + 1], cb[c], cb[c + 1]
                if yB - yA <= 2 * INSET or xB - xA <= 2 * INSET:
                    continue
                cell = img[yA + INSET:yB - INSET, xA + INSET:xB - INSET]
                if cell.size == 0:
                    continue
                if return_geometry:
                    cells_out.append((i + 1, r, c, cell, (xA, yA, xB, yB)))
                else:
                    cells_out.append((i + 1, r, c, cell))
    return cells_out


def sheet_to_cells_teal(img, verbose=False, return_geometry=False):
    """Card geometry derived from teal header bboxes (auto-detected sheet
    print color, see teal_card_bboxes).  See _sheet_to_cells_with_boxes."""
    return _sheet_to_cells_with_boxes(
        img, teal_card_bboxes(img), verbose, return_geometry)


def _frame_bright_extent(gray):
    """Sheet's bright-paper column extent over the whole frame, used to
    seed the column bounds when no header color is detectable."""
    colmean = gray.mean(axis=0).astype(np.uint8)
    thr, _ = cv2.threshold(colmean, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = colmean > max(40, int(thr) * 0.90)
    xs = np.where(bright)[0]
    if xs.size == 0:
        return None
    return int(xs[0]), int(xs[-1])


def forced_card_bboxes(img, band_tops):
    """Build 3 card-band boxes from user-tapped rows (each tap = the TOP
    edge of a card's gray header bar, in the normalized image's pixel
    space).  The card body top is band_top + band height, where band
    height is a fixed fraction of the card pitch (the printed header bars
    are a consistent ~0.13 of card pitch across sheet colors); the column
    extent comes from the card's bright-paper body (see _card_x_extent),
    independent of the washed-out header color."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tops = sorted(int(round(min(H - 2, max(2, float(t)))))
                  for t in band_tops)
    if len(tops) < 3:
        return []
    ext = _frame_bright_extent(gray)
    cx = (ext[0] + ext[1]) / 2 if ext else W / 2
    boxes = []
    for i, t in enumerate(tops):
        pitch = (tops[1] - tops[0]) if i == 0 else (tops[i] - tops[i - 1])
        if i + 1 < len(tops):
            next_t = tops[i + 1]
        else:
            next_t = min(H, t + int(round(1.25 * max(120, pitch))))
        h = int(max(30, min(90, round(0.13 * pitch))))
        y1 = min(H - 2, t + h)
        xL, xR = _card_x_extent(gray, y1 + 2, next_t, cx)
        if xL < 0:
            xL, xR = ext if ext else (0, W - 1)
        boxes.append((int(xL), t, int(xR), int(y1)))
    return boxes


def sheet_to_cells_forced(img, band_tops, verbose=False,
                          return_geometry=False):
    """Card geometry pinned to user-tapped band tops — the assisted-scan
    path for washed-out photos (e.g. gray sheets) where no header color
    gate can find the card bands on its own."""
    return _sheet_to_cells_with_boxes(
        img, forced_card_bboxes(img, band_tops), verbose, return_geometry)


def _card_x_extent(gray, y0, y1, cx):
    """Bright-paper horizontal extent of the card region: among contiguous
    runs of bright columns, pick the one containing cx (the teal band's
    center), preferring it over any background bright region."""
    band = gray[y0:min(y1, gray.shape[0]), :]
    if band.size == 0:
        return -1, -1
    colmean = band.mean(axis=0)
    c8 = colmean.astype(np.uint8)
    thr, _ = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = c8 > max(40, int(thr) * 0.90)
    runs = []
    cur = None
    for x, b in enumerate(bright):
        if b and cur is None:
            cur = [x, x]
        elif b:
            cur[1] = x
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    runs = [r for r in runs if r[1] - r[0] >= 30]
    if not runs:
        return -1, -1
    candidate = runs[0]
    for r in runs[1:]:
        if (abs((r[0] + r[1]) / 2 - cx) < abs((candidate[0] + candidate[1]) / 2 - cx)
                or (r[1] - r[0]) > (candidate[1] - candidate[0]) * 2):
            candidate = r
    return candidate[0], candidate[1]


if __name__ == "__main__":
    out = Path("/tmp/photo_cells")
    out.mkdir(exist_ok=True)
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        cells = sheet_to_cells_photo(img, verbose=True)
        print(f"{p}: {len(cells)} cells")
        for cid, r, c, cell in cells:
            cv2.imwrite(str(out / f"{Path(p).stem}_c{cid}_r{r}c{c}.jpg"), cell)