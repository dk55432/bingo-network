#!/usr/bin/env python3
"""
Extract 5×5 bingo cells from a photographed sheet that may contain
one or more cards stacked vertically on a single paper strip.

Strategy
--------
1. Detect the outer paper strip by locating the dense region of
   horizontal + vertical grid lines (unique to the bingo cards).
2. Perspective-warp the whole strip (nearly axis-aligned in practice).
3. Split the warped strip into individual cards via horizontal gutters
   or equal-height fallback.
4. Extract the regular 5×5 grid from each card.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def order_points(points):
    """Return points in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    result[0] = points[np.argmin(sums)]   # top-left
    result[2] = points[np.argmax(sums)]   # bottom-right
    result[1] = points[np.argmin(diffs)]  # top-right
    result[3] = points[np.argmax(diffs)]  # bottom-left
    return result


def four_point_warp(image, corners, output_size=None):
    """
    Perspective-correct a quadrilateral.
    When output_size is None the destination size is derived from the
    average side lengths of the ordered corners.
    """
    corners = order_points(corners)

    if output_size is None:
        (tl, tr, br, bl) = corners
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        width = max(int(width_a), int(width_b), 1)
        height = max(int(height_a), int(height_b), 1)
        # Cap working resolution while preserving aspect
        scale = min(1400 / max(width, 1), 2000 / max(height, 1), 1.0)
        width = max(int(width * scale), 200)
        height = max(int(height * scale), 200)
    else:
        width, height = output_size

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


# ---------------------------------------------------------------------------
# Sheet (outer paper strip) detection – grid-line based
# ---------------------------------------------------------------------------

def detect_sheet(image, debug_path=None):
    """
    Locate the outer boundary of the bingo paper strip by finding the
    dense vertical band of horizontal/vertical grid lines.

    Returns (ordered_corners, debug_mask) or (None, mask) on failure.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Local contrast normalisation helps with uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Black lines on light paper → binary-inverted adaptive threshold.
    binary = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 8
    )

    # Extract long horizontal and vertical line segments.
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1))
    vert_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 20)))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_k)
    vert  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_k)

    # --- X bounds from the column projection of horizontal lines ----------
    col_score = np.sum(horiz > 0, axis=0).astype(np.float32)
    col_score = cv2.GaussianBlur(col_score.reshape(1, -1), (1, 31), 0).ravel()

    peak = int(col_score.argmax())
    if col_score[peak] < 10:
        return None, horiz   # no meaningful grid detected

    thresh_val = col_score[peak] * 0.25
    high = col_score > thresh_val
    # Take the contiguous run that contains the peak.
    left = peak
    while left > 0 and high[left]:
        left -= 1
    left += 1
    right = peak
    while right < w - 1 and high[right]:
        right += 1

    # Guard against a degenerate band.
    if right - left < w * 0.12:
        return None, horiz

    # --- Y bounds from any line activity inside a slightly wider band -----
    band = slice(max(0, left - 30), min(w, right + 30))
    row_has = (
        (np.sum(horiz[:, band] > 0, axis=1) > 1) |
        (np.sum(vert[:, band]  > 0, axis=1) > 1)
    ).astype(np.uint8)

    # Dilate the 1-D signal (wide 1-D kernel) so neighbouring grid lines
    # and the three cards merge into continuous runs.
    dilate_width = 181  # ~90 px each side
    kernel_1d = np.ones((1, dilate_width), np.uint8)
    row_dil = cv2.dilate(row_has.reshape(1, -1), kernel_1d).ravel()

    # Find the longest contiguous True run.
    runs = []
    start = None
    for i, v in enumerate(row_dil > 0):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(row_dil) - 1, len(row_dil) - start))

    if not runs:
        return None, horiz

    runs.sort(key=lambda r: -r[2])
    top, bottom = runs[0][0], runs[0][1]

    if bottom - top < h * 0.35:
        return None, horiz

    # Axis-aligned corners (the photograph is nearly straight-on).
    # Clipping guarantees we never produce coordinates outside the image.
    left   = int(np.clip(left,   0, w - 1))
    right  = int(np.clip(right,  0, w - 1))
    top    = int(np.clip(top,    0, h - 1))
    bottom = int(np.clip(bottom, 0, h - 1))

    corners = np.array([
        [left,  top],
        [right, top],
        [right, bottom],
        [left,  bottom],
    ], dtype=np.float32)

    ordered = order_points(corners)

    if debug_path:
        debug = image.copy()
        cv2.polylines(debug, [ordered.astype(np.int32)], True, (0, 255, 0), 6)
        labels = ["TL", "TR", "BR", "BL"]
        for pt, lab in zip(ordered, labels):
            p = tuple(pt.astype(int))
            cv2.circle(debug, p, 14, (0, 0, 255), -1)
            cv2.putText(debug, lab, (p[0] + 12, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.imwrite(str(debug_path), debug)

    # Return a useful debug mask (the horizontal lines).
    return ordered, horiz


# ---------------------------------------------------------------------------
# Split warped strip into individual cards
# ---------------------------------------------------------------------------

def split_sheet_into_cards(warped_sheet, number_of_cards=3, gap_ratio=0.015):
    """Equal-height split with a small inter-card gap."""
    h, w = warped_sheet.shape[:2]
    gap = int(h * gap_ratio)
    usable = h - gap * (number_of_cards - 1)
    card_h = usable // number_of_cards

    cards = []
    for i in range(number_of_cards):
        y1 = i * (card_h + gap)
        y2 = y1 + card_h
        y1 = max(0, y1)
        y2 = min(h, y2)
        cards.append(warped_sheet[y1:y2].copy())
    return cards


def split_using_horizontal_gaps(warped_sheet, number_of_cards=3):
    """
    Locate low-variation horizontal bands (the gutters between cards)
    and split on the strongest internal gaps.
    Returns a list of card images or None if detection fails.
    """
    gray = cv2.cvtColor(warped_sheet, cv2.COLOR_BGR2GRAY)
    row_std = np.std(gray.astype(np.float32), axis=1)
    smooth = cv2.GaussianBlur(row_std.reshape(-1, 1), (1, 31), 0).ravel()

    threshold = np.percentile(smooth, 18)
    is_gap = smooth < threshold

    runs = []
    start = None
    for y, g in enumerate(is_gap):
        if g and start is None:
            start = y
        elif not g and start is not None:
            if y - start >= 4:
                runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_gap) - 1))

    if len(runs) < number_of_cards - 1:
        return None

    centers = sorted((a + b) // 2 for a, b in runs)
    h = len(gray)

    from itertools import combinations
    best = None
    for chosen in combinations(centers, number_of_cards - 1):
        cuts = [0] + list(chosen) + [h]
        sizes = [cuts[i + 1] - cuts[i] for i in range(number_of_cards)]
        if min(sizes) < h * 0.12:
            continue
        score = min(sizes)
        if best is None or score > best[0]:
            best = (score, cuts)

    if best is None:
        return None

    _, cuts = best
    return [warped_sheet[cuts[i]:cuts[i + 1]].copy()
            for i in range(number_of_cards)]


def split_using_pink_headers(warped_sheet, number_of_cards=3):
    """
    Split the warped strip by locating the pink BINGO header bands.
    Each card starts at (or just before) its header and ends just
    before the next header (or at the end of content).
    """
    h, w = warped_sheet.shape[:2]
    hsv = cv2.cvtColor(warped_sheet, cv2.COLOR_BGR2HSV)

    # Distinctive pink of the BINGO headers
    pink = (
        ((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 162)) &
        (hsv[:, :, 1] > 45) &
        (hsv[:, :, 2] > 140)
    )
    row_score = np.sum(pink, axis=1).astype(np.float32)
    row_s = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 21), 0).ravel()

    # Strong, well-separated peaks (only the true saturated headers)
    peaks = []
    for i in np.argsort(row_s)[::-1]:
        # Require a high absolute score so weak residual bands are ignored
        if row_s[i] < max(120, row_s.max() * 0.35):
            break
        if all(abs(i - p) > 200 for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= number_of_cards + 1:
            break
    peaks = sorted(peaks)

    if len(peaks) < number_of_cards:
        return None

    headers = peaks[:number_of_cards]

    # End of printable content (avoid dark table at the bottom)
    gray = cv2.cvtColor(warped_sheet, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)
    content_end = h - 1
    for y in range(h - 1, h // 2, -1):
        if row_mean[y] > 70:
            content_end = min(h - 1, y + 10)
            break

    # Build card ranges: start a few pixels before each header
    starts = []
    ends = []
    for i, hy in enumerate(headers):
        start = max(0, hy - 8) if i > 0 else max(0, hy - 15)
        if i + 1 < len(headers):
            end = headers[i + 1] - 5
        else:
            end = content_end
        starts.append(start)
        ends.append(end)

    # Sanity: each card should be a reasonable fraction of the strip
    heights = [e - s for s, e in zip(starts, ends)]
    if min(heights) < h * 0.12:
        return None

    cards = [warped_sheet[s:e].copy() for s, e in zip(starts, ends)]
    return cards


# ---------------------------------------------------------------------------
# Grid / cell extraction
# ---------------------------------------------------------------------------

def _crop_to_number_grid(card):
    """
    Crop to the 5×5 number area.

    Y bounds come from the cluster of strong horizontal grid lines.
    X bounds use a modest fixed side crop (the strip warp already
    centres the paper, so a percentage trim is reliable).
    Returns (roi, y_offset, x_offset).
    """
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    # --- Horizontal lines → top / bottom of the number grid ----------------
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 5), 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    prof = cv2.GaussianBlur(
        horiz.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 5), 0
    ).ravel()

    min_dist = max(12, h // 14)
    peaks = []
    for i in np.argsort(prof)[::-1]:
        if prof[i] < prof.max() * 0.22:
            break
        if all(abs(i - p) >= min_dist for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= 8:
            break
    peaks = sorted(peaks)

    if len(peaks) >= 4:
        med = float(np.median(np.diff(peaks)))
        top = max(0, int(peaks[0] - med * 0.55))
        bottom = min(h, int(peaks[-1] + med * 0.55))
    else:
        top = int(h * 0.14)
        bottom = int(h * 0.94)

    top = int(np.clip(top, 0, h // 3))
    bottom = int(np.clip(bottom, top + h // 4, h))

    # --- Side crop (fixed fraction – reliable after the strip warp) --------
    left = max(4, int(w * 0.04))
    right = w - left

    roi = card[top:bottom, left:right]
    return roi, top, left



def extract_grid_cells(warped, rows=5, columns=5):
    """
    Extract the 5×5 cells from a single card image.

    Horizontal lines: morphological projection (reliable on these cards).
    Vertical lines: cluster digit-ink centroids into `columns` groups and
    place dividers at the midpoints — does not depend on faint printed
    vertical grid lines.

    Returns (cells, x_lines, y_lines) in the card's coordinate system.
    """
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    # ----- Horizontal grid lines ------------------------------------------
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 4), 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    hprof = cv2.GaussianBlur(
        horiz.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 5), 0
    ).ravel()

    exp_h = h * 0.70 / rows
    md = max(10, int(exp_h * 0.50))
    hpeaks = []
    for i in np.argsort(hprof)[::-1]:
        if hprof[i] < hprof.max() * 0.12:
            break
        if all(abs(i - p) >= md for p in hpeaks):
            hpeaks.append(int(i))
        if len(hpeaks) >= rows + 4:
            break
    hpeaks = sorted(hpeaks)

    n = rows + 1
    y_lines = None
    if len(hpeaks) >= n:
        best = None
        for i in range(len(hpeaks) - n + 1):
            sub = hpeaks[i : i + n]
            diffs = np.diff(sub)
            mean_d = float(np.mean(diffs))
            if mean_d < 5:
                continue
            reg = 1.0 / (1.0 + np.std(diffs) / mean_d)
            span = (sub[-1] - sub[0]) / h
            if span < 0.35:
                continue
            score = reg * span
            if best is None or score > best[0]:
                best = (score, list(sub))
        if best is not None:
            y_lines = best[1]

    if y_lines is None:
        y_lines = list(np.linspace(int(h * 0.12), int(h * 0.92), n).astype(int))

    diffs = np.diff(y_lines)
    if len(diffs) >= 3:
        internal = diffs[1:-1] if len(diffs) > 3 else diffs
        med = float(np.median(internal))
        if abs(diffs[-1] - med) > med * 0.12:
            y_lines[-1] = int(round(y_lines[-2] + med))
        if abs(diffs[0] - med) > med * 0.12:
            y_lines[0] = int(round(y_lines[1] - med))
        y_lines = [int(np.clip(y, 0, h - 1)) for y in y_lines]

    # ----- Vertical lines via digit-column clustering ---------------------
    y0 = max(0, y_lines[0] - 5)
    y1 = min(h, y_lines[-1] + 5)
    roi = g[y0:y1, :]
    _, dark = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark, connectivity=8
    )

    digit_cx = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]
        if not (40 < area < 5000 and 6 < bw < 80 and 12 < bh < 100):
            continue
        if cx < 20 or cx > roi.shape[1] - 20:
            continue
        if cy < 5 or cy > roi.shape[0] - 10:
            continue
        digit_cx.append(float(cx))

    x_lines = None
    if len(digit_cx) >= columns * 2:
        digit_cx = np.array(digit_cx, dtype=np.float64)
        # 1-D k-means for column centers
        centers = np.linspace(digit_cx.min(), digit_cx.max(), columns)
        for _ in range(25):
            dists = np.abs(digit_cx[:, None] - centers[None, :])
            lab = dists.argmin(axis=1)
            new_c = np.array([
                digit_cx[lab == k].mean() if np.any(lab == k) else centers[k]
                for k in range(columns)
            ])
            if np.allclose(centers, new_c, atol=0.5):
                break
            centers = new_c
        centers = np.sort(centers)

        gaps = np.diff(centers)
        med_gap = float(np.median(gaps)) if len(gaps) else (w * 0.8 / columns)
        dividers = [0.5 * (centers[i] + centers[i + 1]) for i in range(columns - 1)]
        left = centers[0] - med_gap / 2.0
        right = centers[-1] + med_gap / 2.0
        x_lines = [int(round(left))] + [int(round(d)) for d in dividers] + [int(round(right))]
        x_lines = [int(np.clip(x, 0, w - 1)) for x in x_lines]

        # Ensure strictly increasing
        for i in range(1, len(x_lines)):
            if x_lines[i] <= x_lines[i - 1]:
                x_lines[i] = x_lines[i - 1] + max(10, int(med_gap * 0.4))
        x_lines[-1] = min(x_lines[-1], w - 1)

        # Regularity check — if bad, fall back
        dx = np.diff(x_lines)
        if len(dx) >= 2 and (np.std(dx) / max(np.mean(dx), 1)) > 0.25:
            x_lines = None

    if x_lines is None:
        # Fallback: uniform between outer content bounds
        lefts, rights = [], []
        for y in y_lines:
            yy = int(np.clip(y, 2, h - 3))
            strip = g[yy - 1:yy + 2].min(axis=0)
            dark_idx = np.where(strip < 105)[0]
            if len(dark_idx) > 8:
                lefts.append(int(np.percentile(dark_idx, 3)))
                rights.append(int(np.percentile(dark_idx, 97)))
        left = int(np.median(lefts)) if lefts else int(w * 0.05)
        right = int(np.median(rights)) if rights else int(w * 0.95)
        if right - left < w * 0.4:
            left, right = int(w * 0.05), int(w * 0.95)
        x_lines = list(np.linspace(left, right, columns + 1).astype(int))

    # ----- Extract cells with modest inset to avoid grid-line ink ----------
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(columns):
            x1, x2 = x_lines[c], x_lines[c + 1]
            y1, y2 = y_lines[r], y_lines[r + 1]
            mx = max(3, (x2 - x1) // 10)
            my = max(3, (y2 - y1) // 10)
            cell = warped[y1 + my:y2 - my, x1 + mx:x2 - mx]
            row_cells.append(cell)
        cells.append(row_cells)

    return cells, x_lines, y_lines




def save_cells(cells, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ri, row in enumerate(cells):
        for ci, cell in enumerate(row):
            path = output_dir / f"{prefix}_r{ri + 1}_c{ci + 1}.png"
            cv2.imwrite(str(path), cell)


def save_corner_debug(image, corners, filename):
    debug = image.copy()
    corners = order_points(corners).astype(np.int32)
    cv2.polylines(debug, [corners], True, (0, 0, 255), 6)
    for pt, lab in zip(corners, ["TL", "TR", "BR", "BL"]):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(debug, (x, y), 12, (255, 0, 0), -1)
        cv2.putText(debug, lab, (x + 14, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
    cv2.imwrite(str(filename), debug)


def draw_grid_debug(card, x_lines, y_lines, filename):
    debug = card.copy()
    for x in x_lines:
        cv2.line(debug, (int(x), 0), (int(x), debug.shape[0] - 1), (0, 0, 255), 2)
    for y in y_lines:
        cv2.line(debug, (0, int(y)), (debug.shape[1] - 1, int(y)), (0, 0, 255), 2)
    cv2.imwrite(str(filename), debug)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(image_path, output_dir="output", number_of_cards=3):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    stem = image_path.stem

    # ---- 1. Detect outer strip ------------------------------------------------
    sheet_corners, debug_mask = detect_sheet(
        image,
        debug_path=output_dir / f"{stem}_corners.png"
    )
    cv2.imwrite(str(output_dir / "debug_grid_lines.png"), debug_mask)

    if sheet_corners is None:
        raise RuntimeError("Could not detect the outer sheet boundary")

    save_corner_debug(image, sheet_corners,
                      output_dir / f"{stem}_corners_labelled.png")

    # ---- 2. Warp the whole strip ---------------------------------------------
    warped_sheet = four_point_warp(image, sheet_corners)
    cv2.imwrite(str(output_dir / f"{stem}_warped_sheet.png"), warped_sheet)

    # ---- 3. Split into individual cards --------------------------------------
    cards = split_using_pink_headers(warped_sheet, number_of_cards)
    if cards is None or len(cards) != number_of_cards:
        print("Pink-header split failed – trying gap detection")
        cards = split_using_horizontal_gaps(warped_sheet, number_of_cards)
    if cards is None or len(cards) != number_of_cards:
        print("Gap detection failed – falling back to equal-height split")
        cards = split_sheet_into_cards(warped_sheet, number_of_cards)

    # ---- 4. Extract 5×5 cells from each card ---------------------------------
    for idx, card in enumerate(cards):
        card_name = f"card_{idx + 1}"
        cv2.imwrite(str(output_dir / f"{card_name}.png"), card)

        cells, x_lines, y_lines = extract_grid_cells(card)
        save_cells(cells, output_dir / "cells", card_name)
        draw_grid_debug(card, x_lines, y_lines,
                        output_dir / f"{card_name}_grid_debug.png")

        print(f"  {card_name}: extracted {len(cells)}×{len(cells[0])} cells")

    print(f"Done. Results written under: {output_dir.resolve()}")
    return cards


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_file = "bingo_sheet.jpg"
    else:
        image_file = sys.argv[1]

    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    process_image(image_file, output_dir=out)
