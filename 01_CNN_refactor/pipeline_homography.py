"""
Homography-based pipeline for bingo card reading.

Uses SIFT feature matching + homography to detect bingo card headers
(using color-variant templates), then projects precisely-measured column
boundaries from template space into the scene. Row boundaries use the
proven dip+snap method from the original pipeline.

This is an alternative to the Hough-based column detection in pipeline.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Template configuration for header color variants
# Each template should be a clean flatbed scan of just the BINGO header strip
TEMPLATE_DIR = Path(__file__).parent / "header_templates"

# Column boundaries as fractions of template width (measured on clean scans)
COLUMN_BOUNDARIES_X_FRAC = {
    "B|I": 0.19,
    "I|N": 0.385,
    "N|G": 0.57,
    "G|O": 0.75,
}
TOP_Y_FRAC = 0.0
BOTTOM_Y_FRAC = 1.0

# SIFT matching parameters
RATIO_TEST_THRESHOLD = 0.75
MIN_GOOD_MATCHES = 12
RANSAC_REPROJ_THRESHOLD = 5.0
MAX_CARDS = 3

GRID_SIZE = 5
INSET = 5


def _load_template_features(detector, template_paths):
    """Extract keypoints/descriptors for every color-variant template."""
    templates = []
    for path in template_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read template: {path}")
        kp, des = detector.detectAndCompute(img, None)
        h, w = img.shape[:2]
        templates.append({
            "path": path,
            "img": img,
            "kp": kp,
            "des": des,
            "w": w,
            "h": h,
        })
    return templates


def _template_boundary_points(template):
    """Return the boundary points (top & bottom) in template pixel space."""
    w, h = template["w"], template["h"]
    pts = {}
    for name, xf in COLUMN_BOUNDARIES_X_FRAC.items():
        top = (xf * w, TOP_Y_FRAC * h)
        bot = (xf * w, BOTTOM_Y_FRAC * h)
        pts[name] = (top, bot)
    return pts


def _find_best_instance(matcher, template, scene_kp, scene_des, active_mask):
    """
    Match template descriptors against the currently-active subset of scene
    keypoints/descriptors, run RANSAC, and return the homography + inlier
    scene keypoint indices if a confident match is found, else None.
    """
    active_indices = np.where(active_mask)[0]
    if len(active_indices) < MIN_GOOD_MATCHES:
        return None

    sub_des = scene_des[active_indices]
    matches = matcher.knnMatch(template["des"], sub_des, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([template["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # map back from sub-index to original scene keypoint index
    dst_pts = np.float32(
        [scene_kp[active_indices[m.trainIdx]].pt for m in good]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() < MIN_GOOD_MATCHES:
        return None

    inlier_scene_indices = active_indices[[m.trainIdx for i, m in enumerate(good) if inlier_mask[i]]]

    return {
        "H": H,
        "num_inliers": int(inlier_mask.sum()),
        "inlier_scene_indices": inlier_scene_indices,
    }


# ---------------------------------------------------------------------------
# Row processing (borrowed from pipeline.py)
# ---------------------------------------------------------------------------

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


def _trim_grid_bottom(card_img, min_density_frac=0.15):
    """Trim blank space below a card's actual number grid.

    Scan bottom-up for a sustained near-zero density gap (blank paper).
    The actual grid bottom is the last row before that gap with significant
    ink density.  This correctly ignores both blank paper and border lines
    that ramp up at the card edge."""
    if card_img.size == 0 or card_img.shape[0] == 0 or card_img.shape[1] == 0:
        return card_img
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    dark = gray < 128
    row_density = dark.mean(axis=1)
    height = card_img.shape[0]
    if row_density.max() == 0:
        return card_img

    # Scan bottom-up: find the last row that has near-zero density
    # (start of blank region), then find the row before it with real content.
    # Use a rolling window to be robust to single-row noise.
    window = 5
    bottom = height

    # First, find any sustained blank region (10+ consecutive rows < 0.01)
    blank_start = -1
    blank_count = 0
    for y in range(height - 1, -1, -1):
        if row_density[y] < 0.01:
            blank_count += 1
            if blank_count >= 10:
                blank_start = y
        else:
            blank_count = 0

    if blank_start >= 0:
        # Found a blank region; grid bottom is the last dense row before it
        for y in range(blank_start - 1, -1, -1):
            if row_density[y] > 0.05:
                bottom = y + 1
                break
    # else: no blank region found, use full card height

    return card_img[:bottom, :]


def _split_into_cards(warped, min_band_height_frac=0.03):
    """Split a warped strip into one cropped image per card's number grid."""
    # Find header bands using saturation (color-agnostic)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    row_saturation = hsv[:, :, 1].astype(np.float32).mean(axis=1)
    threshold = np.median(row_saturation) + 20
    is_header_row = row_saturation > threshold

    height = warped.shape[0]
    min_band_height = max(1, int(height * min_band_height_frac))

    runs = []
    in_band = False
    band_start = 0
    for y, flag in enumerate(is_header_row):
        if flag and not in_band:
            in_band, band_start = True, y
        elif not flag and in_band:
            in_band = False
            if y - band_start >= min_band_height:
                runs.append((band_start, y))
    if in_band and height - band_start >= min_band_height:
        runs.append((band_start, height))

    if not runs:
        return [warped]

    regions = []
    for i, (h_start, h_end) in enumerate(runs):
        grid_top = h_end
        grid_bottom = runs[i + 1][0] if i + 1 < len(runs) else height
        if grid_bottom - grid_top >= height * 0.08:
            regions.append((grid_top, grid_bottom))

    leading_height = runs[0][0]
    if regions:
        typical_height = sum(b - a for a, b in regions) / len(regions)
        if leading_height >= typical_height * 0.85:
            regions.insert(0, (0, leading_height))

    if not regions:
        regions = [(0, height)]

    cards = [warped[top:bottom, :] for top, bottom in regions]
    return [_trim_grid_bottom(card) for card in cards]


def _extract_rows_from_card(card_img, x0, x1):
    """Extract 5 row boundaries from a single card image.

    Printed bingo cards have exactly 5 equally-spaced rows below the header.
    We detect the paper extent (excluding header-band color bleed and dark
    footer) using a relative brightness threshold, then divide that region
    into 5 equal rows."""
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    dark = gray < 128
    row_density = dark.mean(axis=1)
    row_median = np.median(gray, axis=1)

    # Relative threshold: 40% of the card's brightest row median.
    # This adapts to different lighting conditions.
    bright_pct = np.percentile(row_median, 90)
    paper_threshold = max(40, bright_pct * 0.4)

    # Find grid_start: the start of the number grid, which is the first run of
    # consecutive rows with real ink content.  A blank top margin is bright but
    # has very low density; it must NOT be counted as grid content (it would
    # inflate every row).  Require 3+ consecutive rows above a modest density
    # threshold so single-row noise can't trigger it.
    grid_start = 0
    for y in range(h - 2):
        if row_density[y] > 0.09 and row_density[y + 1] > 0.09 and row_density[y + 2] > 0.09:
            grid_start = y
            break

    # Find last row above threshold
    grid_end = h
    for y in range(h - 1, -1, -1):
        if row_median[y] > paper_threshold:
            grid_end = y + 1
            break

    grid_h = grid_end - grid_start
    if grid_h < h * 0.5:
        # Fallback: paper detection failed, divide full card
        return [h * k // 5 for k in range(6)]

    return [grid_start + grid_h * k // 5 for k in range(6)]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _load_templates(detector):
    """Load all header template images from the template directory."""
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template directory not found: {TEMPLATE_DIR}")
    
    template_paths = list(TEMPLATE_DIR.glob("*.jpg")) + list(TEMPLATE_DIR.glob("*.png"))
    if not template_paths:
        raise FileNotFoundError(f"No template images found in {TEMPLATE_DIR}")
    return _load_template_features(detector, template_paths)


def detect_cards_homography(img, verbose=False):
    """
    Detect cards in a full sheet image using a hybrid approach:
    1. Use HSV color detection (teal_card_bboxes) to find all header bands reliably
    2. For each band, run SIFT matching (for verification + diagnostics)
    3. Slice the card grids with the proven photo_sheet_cells geometry:
       equal division on a uniform lattice, snapped to the printed dark
       grid lines (dip peaks/Hough).  Template-corner projection does NOT
       yield the true grid width (the header template often matches a
       sub-region of the band), so SIFT is not used to place columns.

    Returns list of dicts with card_img, col_bounds, row_bounds, etc.
    """
    from photo_sheet_cells import (
        _sheet_to_cells_with_boxes,
        teal_card_bboxes,
    )
    from cnn_reader import _normalize, _scale_boxes

    # Step 1: detect all header bands via HSV color, on the ORIGINAL image
    # (normalization changes color statistics and can break teal detection).
    header_boxes = teal_card_bboxes(img)
    # Mirror the CNN reader: when fewer than 3 bands are found on the
    # original, retry on the normalized image (it can reveal faded headers).
    norm_img = _normalize(img)
    if header_boxes and len(header_boxes) < 3:
        norm_boxes = teal_card_bboxes(norm_img)
        if len(norm_boxes) > len(header_boxes):
            header_boxes = _scale_boxes(norm_boxes, norm_img.shape, img.shape)
    if verbose:
        print(f"Teal detected {len(header_boxes)} header band(s)")
    if not header_boxes:
        return []

    # Step 2: SIFT-verify each band (diagnostic; does not drive geometry).
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    templates = _load_templates(detector)
    scene_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scene_kp, scene_des = detector.detectAndCompute(scene_gray, None)
    if verbose and (scene_des is not None):
        for band_idx, (_, by0, _, by1) in enumerate(header_boxes):
            band_h = by1 - by0
            margin = band_h * 2
            y_lo = max(0, by0 - margin)
            y_hi = min(img.shape[0], by1 + margin)
            y_coords = np.array([kp.pt[1] for kp in scene_kp])
            band_mask = (y_coords >= y_lo) & (y_coords <= y_hi)
            best_inliers = 0
            best_name = None
            for template in templates:
                result = _find_best_instance(matcher, template,
                                             scene_kp, scene_des, band_mask)
                if result and result["num_inliers"] > best_inliers:
                    best_inliers = result["num_inliers"]
                    best_name = template["path"].name
            print(f"  Band {band_idx} (y={by0}-{by1}): SIFT match "
                  f"{best_name or 'none'} ({best_inliers} inliers)")

    # Step 3: slice each card's grid with the proven grid fitter on the
    # NORMALIZED image (same cell content the CNN reader model was trained
    # on).  This equal-divides card regions (clamped to the paper bright
    # extent) and snaps each boundary to the printed grid lines, mirroring
    # the CNN reader's geometry (photo_sheet_cells._sheet_to_cells_with_boxes).
    scaled = _scale_boxes(header_boxes, img.shape, norm_img.shape)
    items = _sheet_to_cells_with_boxes(
        norm_img, scaled, verbose=verbose, return_geometry=True)
    by = {}
    for cid, r, c, cell, box in items:
        by.setdefault(cid, {})[(r, c)] = box

    results = []
    for cid in sorted(by):
        cells = by[cid]
        # Reconstruct the 6x6 boundary grid from cell rects.
        col_bounds = [min(cells[(r, c)][0] for r in range(5)) for c in range(5)]
        col_bounds.append(max(cells[(r, 4)][2] for r in range(5)))
        row_bounds = [min(cells[(r, c)][1] for c in range(5)) for r in range(5)]
        row_bounds.append(max(cells[(4, c)][3] for c in range(5)))
        card_left, card_right = col_bounds[0], col_bounds[-1]
        card_top, card_bottom = row_bounds[0], row_bounds[-1]

        card_img = norm_img[card_top:card_bottom, card_left:card_right]
        if card_img.size == 0:
            continue

        if verbose:
            print(f"    card{cid}: y=[{card_top},{card_bottom}] "
                  f"x=[{card_left},{card_right}] "
                  f"cols={[b - card_left for b in col_bounds]} "
                  f"rows={[b - card_top for b in row_bounds]}")

        results.append({
            "card_img": card_img,
            "col_bounds": [b - card_left for b in col_bounds],
            "row_bounds": [b - card_top for b in row_bounds],
            "card_top": card_top,
            "card_bottom": card_bottom,
            "card_left": card_left,
            "card_right": card_right,
        })

    return results


def _project_boundaries(instance):
    template = instance["template"]
    H = instance["H"]
    boundary_pts_template = _template_boundary_points(template)
    projected = {}
    for name, (top, bot) in boundary_pts_template.items():
        pts = np.float32([top, bot]).reshape(-1, 1, 2)
        scene_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        projected[name] = {"top": tuple(scene_pts[0]), "bottom": tuple(scene_pts[1])}
    return projected


def _sort_instances_top_to_bottom(instances, projections):
    def y_key(item):
        _, proj = item
        return proj["B|I"]["top"][1]
    return sorted(zip(instances, projections), key=y_key)


def _build_full_column_lines(ordered_projections, scene_height):
    boundary_names = list(COLUMN_BOUNDARIES_X_FRAC.keys())
    lines = {}
    for name in boundary_names:
        top_points = [proj[name]["top"] for _, proj in ordered_projections]
        bottom_points = [proj[name]["bottom"] for _, proj in ordered_projections]
        polyline = []
        for t, b in zip(top_points, bottom_points):
            polyline.append(t)
            polyline.append(b)
        lines[name] = {"polyline": polyline}
        if len(polyline) >= 2:
            (x0, y0), (x1, y1) = polyline[0], polyline[1]
            slope = (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else 0
            top_extrap_y = 0
            top_extrap_x = x0 - slope * (y0 - top_extrap_y)
            lines[name]["extrapolated_top"] = (top_extrap_x, top_extrap_y)
            (x0, y0), (x1, y1) = polyline[-2], polyline[-1]
            slope = (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else 0
            bot_extrap_y = scene_height
            bot_extrap_x = x1 + slope * (bot_extrap_y - y1)
            lines[name]["extrapolated_bottom"] = (bot_extrap_x, bot_extrap_y)
    return lines


def segment_card_homography(card_img, col_bounds, row_bounds):
    """Slice a card into 5x5 cells using given boundaries."""
    cells = []
    for r in range(GRID_SIZE):
        row_cells = []
        y0, y1 = row_bounds[r], row_bounds[r + 1]
        for c in range(GRID_SIZE):
            x0, x1 = col_bounds[c], col_bounds[c + 1]
            cell = card_img[y0 + INSET:y1 - INSET, x0 + INSET:x1 - INSET]
            if cell.size > 0:
                row_cells.append(cell)
            else:
                row_cells.append(np.zeros((28, 28), dtype=np.uint8))
        cells.append(row_cells)
    return cells


def read_sheet_homography(img, verbose=False):
    """
    Main entry point: process a full sheet image through the homography pipeline.
    Returns list of dicts with 5x5 cell grids, matching the existing pipeline's format.
    """
    # Load CNN model for cell recognition
    from cnn_reader import _normalize, load_model, cell_logits, decode_card
    import time
    model = load_model()

    norm_img = _normalize(img)  # header detection still runs on raw img
    results = detect_cards_homography(img, verbose)
    output = []
    
    # Create debug directory and save overlay
    ts = int(time.time())
    dbg = Path("/tmp/cnn_reader_debug")
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / f"in_{ts}.png"), norm_img)
    
    # Draw overlay with column boundaries (normalized-image coordinates,
    # matching the in_ dump the CNN reader writes)
    overlay = norm_img.copy()
    
    for res in results:
        cells = segment_card_homography(res["card_img"], res["col_bounds"], res["row_bounds"])
        # Run CNN on each cell
        lgrid = {}
        for r in range(5):
            for c in range(5):
                lgrid[(r, c)] = cell_logits(model, cells[r][c])
        # Decode with column constraints
        numbers, confs = decode_card(lgrid)
        # Format to match existing pipeline output
        grid = []
        for r in range(5):
            row = []
            for c in range(5):
                if r == 2 and c == 2:
                    row.append({"value": None, "is_free_space": True, "valid": True, "raw": "FREE", "confidence": 100.0})
                else:
                    v = numbers[r][c]
                    row.append({"value": v, "is_free_space": False, "valid": v is not None, "raw": str(v) if v is not None else None, "confidence": round(100 * confs[r][c], 1)})
            grid.append(row)
        
        # Draw green overlay for this card
        card_left = res["card_left"]
        card_top = res["card_top"]
        col_bounds = res["col_bounds"]
        row_bounds = res["row_bounds"]
        for r in range(5):
            for c in range(5):
                x0 = card_left + col_bounds[c]
                y0 = card_top + row_bounds[r]
                x1 = card_left + col_bounds[c + 1]
                y1 = card_top + row_bounds[r + 1]
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 0), 2)
        
        output.append({
            "grid": grid,
            "col_bounds": res["col_bounds"],
            "row_bounds": res["row_bounds"],
            "card_bbox": (res["card_left"], res["card_top"], res["card_right"], res["card_bottom"])
        })
    
    cv2.imwrite(str(dbg / f"overlay_{ts}.png"), overlay)
    return output, ts


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline_homography.py <image_path>")
        sys.exit(1)
    
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)
    
    results, ts = read_sheet_homography(img, verbose=True)
    print(f"Detected {len(results)} cards")
    for i, r in enumerate(results):
        print(f"  Card {i+1}: {r['card_bbox']}, cols={r['col_bounds']}, rows={r['row_bounds']}")