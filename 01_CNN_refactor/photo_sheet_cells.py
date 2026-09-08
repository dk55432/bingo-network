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
    find_header_bands,
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Frame-level paper extent (hard constraint - paper can't be wider than this)
    frame_ext = _frame_bright_extent(gray)
    for y0, y1 in merged:
        cols = np.where(teal[y0:y1 + 1, :].any(axis=0))[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])

        # Clamp horizontal extent to the paper's bright-paper region.
        # The teal header color can bleed/reflect onto the desk, making
        # the mask wider than the actual card. Use the card body region
        # (just below the header) to find the true paper edges.
        body_y0 = min(y1 + 2, gray.shape[0] - 1)
        body_y1 = min(y1 + 60, gray.shape[0])
        if body_y1 > body_y0:
            body_band = gray[body_y0:body_y1, :]
            colmean = body_band.mean(axis=0).astype(np.uint8)
            thr, _ = cv2.threshold(colmean, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright = colmean > max(40, int(thr) * 0.90)
            xs = np.where(bright)[0]
            if xs.size > 0:
                x0 = max(x0, int(xs[0]))
                x1 = min(x1, int(xs[-1]))

        # Also clamp to frame-level paper extent as a hard constraint.
        # The header mask can bleed onto the desk; the full-frame paper
        # detection is more reliable for the absolute outer bounds.
        if frame_ext is not None:
            fx0, fx1 = frame_ext
            x0 = max(x0, fx0)
            x1 = min(x1, fx1)

        boxes.append((x0, y0, x1, y1))
    boxes = sorted(boxes, key=lambda b: b[1])
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


def _grid_dip_v(gray, x0, x1, top, bottom, strip_h=None):
    """Per-column brightness-dip score across the full card height.
    Vertical grid lines are continuous dark lines spanning the card;
    digit strokes are localized to rows. Averaging over full height
    suppresses digit strokes while preserving grid lines."""
    W = gray.shape[1]
    if strip_h is None:
        yL, yR = top, bottom  # full card height
    else:
        ym = (top + bottom) // 2
        yL = max(0, ym - strip_h // 2)
        yR = min(gray.shape[0], ym + strip_h // 2)
    strip = gray[yL:yR, x0:x1].mean(axis=0).astype(np.float64)
    n = len(strip)
    score = np.zeros(n)
    for j in range(5, n - 5):
        neighbours = np.concatenate([strip[j - 5:j - 1], strip[j + 2:j + 6]])
        score[j] = neighbours.mean() - strip[j]
    full = np.zeros(W)
    full[x0:x0 + n] = score
    return full


def _find_grid_lines_phone(card_img, verbose=False):
    """Phone-photo variant of find_grid_line_positions with relaxed params.
    Phone photos have fainter grid lines; lower thresholds and shorter min length.
    """
    from pipeline import _cluster_1d, _select_best_lines
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # Lower Canny thresholds for fainter lines
    edges = cv2.Canny(gray, 30, 100)
    h, w = edges.shape

    # More permissive: shorter min length (8% vs 15%), lower Hough threshold
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30,
        minLineLength=int(w * 0.08), maxLineGap=50,
    )

    horiz_ys, vert_xs = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5
            if abs(dy) <= 5 and length >= w * 0.08:
                horiz_ys.append((y1 + y2) // 2)
            elif abs(dx) <= 5 and length >= h * 0.08:
                vert_xs.append((x1 + x2) // 2)

    horiz_clusters = _cluster_1d(horiz_ys, tol=h * 0.02)
    vert_clusters = _cluster_1d(vert_xs, tol=w * 0.02)

    h_margin = h * 0.03
    w_margin = w * 0.03
    horiz_clusters = [(y, c) for y, c in horiz_clusters if h_margin < y < h - h_margin]
    vert_clusters = [(x, c) for x, c in vert_clusters if w_margin < x < w - w_margin]

    horiz_ys = _select_best_lines(horiz_clusters)
    vert_xs = _select_best_lines(vert_clusters)
    if verbose:
        print(f"  phone Hough: h={len(horiz_ys)} v={len(vert_xs)} lines")
    return horiz_ys, vert_xs


def _sheet_to_cells_with_boxes(
        img, boxes, verbose=False, return_geometry=False):
    """Shared per-card cell extraction.  Rows use the proven brightness-dip
    + snap (robust on normalized phone images).  Columns use pipeline Hough
    on the full card width (from frame bright extent) + uniform lattice fit."""
    if not boxes:
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get frame-level paper extent once from grid regions below headers.
    # This gives the true horizontal bounds of the card grid.
    frame_ext = _frame_bright_extent(gray, boxes)
    if frame_ext is not None:
        grid_x0, grid_x1 = frame_ext
    else:
        grid_x0, grid_x1 = 0, W
    
    cells_out = []
    for i, (hx0, y0, hx1, y1) in enumerate(boxes):
        top = y1 + 2
        bottom = boxes[i + 1][1] if i + 1 < len(boxes) else H
        if i + 1 == len(boxes):
            bottom = _card_y_extent(gray, top, bottom)
        if bottom - top < 60 or grid_x1 - grid_x0 < 60:
            if verbose:
                print(f"card{i + 1}: region too thin ({bottom - top}x{grid_x1 - grid_x0})")
            continue
        
        # Use header box for vertical position (y), but frame extent for horizontal (x)
        x0, x1 = grid_x0, grid_x1

# ---- ROWS: proven dip + snap ----
        grid_sig = _grid_dip(gray, x0, x1, top, bottom)
        rb = [top + (bottom - top) * k // 5 for k in range(6)]
        rb = _snap(rb, grid_sig, top, bottom, radius=40)

        # ---- COLUMNS: equal division with tight snapping to detected peaks ----
        # Use equal division as the strong prior. Only snap to detected peaks
        # if they are very close to the expected positions. This avoids the
        # sequential-snapping error propagation when dividers are missing.
        card_crop = img[top:bottom, x0:x1]
        _, vert_xs = find_grid_line_positions(card_crop)
        width = x1 - x0
        
        # Start with perfect equal division
        cb_rel = [round(width * k / 5) for k in range(6)]
        
        # Only snap to a detected peak if it's very close to the expected position
        # (tight tolerance: 12% of cell width, ~17px for typical 144px pitch)
        tol = max(10, int(0.12 * width / 5))
        for k in range(1, 5):
            expected = round(width * k / 5)
            # Search in both ALL detected peaks (including edges)
            candidates = [p for p in vert_xs if abs(p - expected) <= tol]
            if candidates:
                cb_rel[k] = min(candidates, key=lambda p: abs(p - expected))
        cb = [x0 + x for x in cb_rel]

        # ---- ROWS: dip + snap, then enforce uniform lattice on snapped boundaries ----
        # (snap already done above for rb)
        # Fit uniform lattice to the 4 snapped interior boundaries to correct
        # any single-boundary mis-snap (e.g. merged rows)
        if len(rb) == 6:
            idx = np.array([0, 1, 2, 3, 4, 5], float)
            a, b = np.polyfit(idx, np.array(rb, float), 1)
            pitch_r = a
            pitch_r = max(70, min(120, pitch_r))
            rb = [int(round(b + a * k)) for k in range(6)]

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


def _frame_bright_extent(gray, header_boxes=None):
    """Sheet's bright-paper column extent over the frame, used to
    seed the column bounds when no header color is detectable.

    If header_boxes are provided, uses their union to define the search range,
    then analyzes the middle third of the image within that range."""
    h = gray.shape[0]
    
    if header_boxes is not None and len(header_boxes) > 0:
        header_x0 = min(b[0] for b in header_boxes)
        header_x1 = max(b[2] for b in header_boxes)
        
        # Analyze the middle third of the image within the union range
        y0, y1 = h // 3, 2 * h // 3
        band = gray[y0:y1, header_x0:header_x1]
        colmean = band.mean(axis=0).astype(np.uint8)
        thr, _ = cv2.threshold(colmean, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = colmean > max(40, int(thr) * 0.90)
        xs = np.where(bright)[0]
        if xs.size > 0:
            return header_x0 + xs[0], header_x0 + xs[-1]
    
    # Fallback: middle third of full image
    y0, y1 = h // 3, 2 * h // 3
    band = gray[y0:y1, :]
    colmean = band.mean(axis=0).astype(np.uint8)
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