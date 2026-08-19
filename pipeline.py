"""
Pure image-processing + OCR pipeline for bingo cards. No FastAPI dependency
here on purpose — this module gets imported both by the running API
(bingo_scan.py) and by the standalone debug_scan.py CLI tool.
"""

import logging

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

GRID_SIZE = 5
WARPED_SIDE = 500  # pixel size of the flattened square card image

# Standard 75-ball bingo column ranges, in card column order.
COLUMN_RANGES = {
    0: (1, 15),    # B
    1: (16, 30),   # I
    2: (31, 45),   # N (center cell is a free space, handled separately)
    3: (46, 60),   # G
    4: (61, 75),   # O
}


class CardNotFoundError(Exception):
    """Raised when no plausible card outline could be found in the image."""


def load_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right: smallest x-y
    rect[3] = pts[np.argmax(diff)]   # bottom-left: largest x-y
    return rect


def find_card_contour(img: np.ndarray, debug: dict | None = None) -> np.ndarray:
    """
    Locate the largest plausible 4-sided contour in the image, assumed to be
    the card. If `debug` is passed, intermediate images are stashed into it
    under 'edges' and 'contour_overlay' for inspection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)

    if debug is not None:
        debug["edges"] = edged

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = img.shape[0] * img.shape[1]
    found = None
    for c in contours[:10]:
        area = cv2.contourArea(c)
        # Reject tiny contours (noise) and near-full-frame contours (the
        # photo's own border, not the card) — a real card photo should be a
        # meaningful chunk of the frame but rarely the entire frame.
        if area < 0.1 * img_area or area > 0.98 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            found = approx.reshape(4, 2)
            break

    if debug is not None:
        overlay = img.copy()
        if found is not None:
            cv2.drawContours(overlay, [found.astype(int)], -1, (0, 0, 255), 4)
        debug["contour_overlay"] = overlay

    if found is None:
        raise CardNotFoundError(
            "Could not find a clear card outline. Retake the photo with better contrast/lighting, "
            "or make sure the card fills most of the frame with a plain background behind it."
        )
    return found


def warp_card(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Perspective-warp the selected region to a flat, top-down view —
    preserving its actual aspect ratio, not forcing a square. Forcing a
    square silently distorts anything that isn't already square (most
    obviously a strip of several stacked cards, but even a single card
    photographed at a slight angle isn't perfectly square), which then
    compounds into misaligned cell boundaries downstream.
    """
    rect = order_points(corners.astype("float32"))
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (max_width, max_height))


def _find_header_bands(warped: np.ndarray, min_band_height_frac: float = 0.03) -> list[tuple[int, int]]:
    """Find horizontal bands that are meaningfully more saturated (i.e.
    colored) than the rest of the image — the header band above each
    card's number grid. Keyed off saturation rather than a specific hue
    (yellow, pink, etc.) so it isn't tied to one card design."""
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
    return runs


def find_card_grid_regions(
    warped: np.ndarray, min_band_height_frac: float = 0.03, min_region_height_frac: float = 0.08
) -> list[tuple[int, int]]:
    """
    Locate the 5x5 number-grid portion of each card on a (possibly
    multi-card) warped strip. Each detected header band marks where a
    new card starts; a card's grid runs from the end of its own header
    to the start of the next card's header (or the image bottom, for
    the last card).

    Works for any number of stacked cards — including exactly one: if no
    header band is detected at all (e.g. an already-cropped single-card
    photo with no colored header), the whole image is returned as a
    single region, matching the old single-card behavior.

    Regions shorter than min_region_height_frac of the total warped
    height are discarded as implausible — a real card's grid should
    occupy a substantial portion of the strip; a region only a few
    pixels tall means a header got detected somewhere it shouldn't have
    (e.g. a shadow or background sliver mistaken for a colored band).
    """
    header_runs = _find_header_bands(warped, min_band_height_frac)
    height = warped.shape[0]
    min_region_height = height * min_region_height_frac

    if not header_runs:
        return [(0, height)]

    # Regions strictly following a detected header — these are trusted
    # completely, same as before.
    trailing_regions = []
    for i, (h_start, h_end) in enumerate(header_runs):
        grid_top = h_end
        grid_bottom = header_runs[i + 1][0] if i + 1 < len(header_runs) else height
        if grid_bottom - grid_top >= min_region_height:
            trailing_regions.append((grid_top, grid_bottom))
        # else: implausibly thin region (e.g. a header detected right
        # next to another, or right at the image's bottom edge) —
        # discard rather than returning a nonsensical sliver.

    regions = list(trailing_regions)

    # Content BEFORE the very first detected header — previously always
    # silently discarded, even when it was a real card's worth of rows
    # (e.g. the crop started mid-card, with that card's own header just
    # off-frame above). Only include it if it's close in height to the
    # other detected cards on this strip: segment_grid always divides a
    # region into exactly GRID_SIZE rows, so feeding it a genuinely
    # incomplete card (fewer real rows than that) would corrupt every
    # row, not just the missing one — worse than just skipping it. With
    # no other regions to compare against, fall back to the general
    # min_region_height_frac threshold.
    leading_height = header_runs[0][0]
    if trailing_regions:
        typical_height = sum(b - a for a, b in trailing_regions) / len(trailing_regions)
        looks_complete = leading_height >= typical_height * 0.85
    else:
        typical_height = None
        looks_complete = leading_height >= min_region_height

    if looks_complete:
        regions.insert(0, (0, leading_height))
    elif leading_height >= min_region_height:
        logger.warning(
            f"find_card_grid_regions: {leading_height}px of content before the "
            f"first detected header looks like an incomplete card (other cards "
            f"on this strip are ~{typical_height:.0f}px) — skipped rather than "
            f"guessed at. If this strip really has another card above what was "
            f"captured, the crop/corners likely need to start higher."
        )

    if not regions:
        # Every candidate region was implausible — fall back to treating
        # the whole image as one card, same as the "no header detected"
        # case above, rather than returning nothing useful at all.
        return [(0, height)]
    return regions


def trim_grid_bottom(card_img: np.ndarray, min_density_frac: float = 0.15) -> np.ndarray:
    """
    Trim any near-blank space below a card's actual number grid (e.g. a
    small footer/serial code plus the gap before the next card on a
    strip) before dividing into GRID_SIZE equal rows. Left in, that
    extra space inflates the "divide by 5" math and drifts every row
    boundary further off the further down the card you go — this is
    what let a footer end up mixed into segment_grid()'s equal-height
    division. Works by scanning up from the bottom for the last row
    with real ink density (a full row of bold digits + grid lines is
    much denser than a thin footer line or blank gap).
    """
    if card_img.size == 0 or card_img.shape[0] == 0 or card_img.shape[1] == 0:
        return card_img  # nothing to trim on an empty/degenerate crop

    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    dark = gray < 128
    row_density = dark.mean(axis=1)
    height = card_img.shape[0]
    if row_density.max() == 0:
        return card_img  # blank image, nothing to trim

    threshold = row_density.max() * min_density_frac
    bottom = height
    for y in range(height - 1, -1, -1):
        if row_density[y] > threshold:
            bottom = y + 1
            break
    return card_img[:bottom, :]


def split_into_cards(warped: np.ndarray, min_band_height_frac: float = 0.03) -> list[np.ndarray]:
    """Split a warped strip into one cropped image per card's number
    grid (header/gap already excluded), each ready to hand to
    segment_grid(). Handles any number of cards, including one. Each
    card is also trimmed of any leftover footer/gap space below its
    actual grid (see trim_grid_bottom) before being returned."""
    regions = find_card_grid_regions(warped, min_band_height_frac)
    cards = [warped[top:bottom, :] for top, bottom in regions]
    return [trim_grid_bottom(card) for card in cards]



def _cluster_1d(values: list[int], tol: float) -> list[tuple[int, int]]:
    """Merge nearby values (e.g. multiple Hough segments detected along
    the same physical line, since a real line gets broken into several
    pieces by the FREE cell's sub-text) into one representative position
    each, alongside how many raw values supported it — a real grid line
    should accumulate many nearby detections; a stray digit-stroke
    artifact should only produce one or two."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(int(sum(c) / len(c)), len(c)) for c in clusters]


def _select_best_lines(clusters: list[tuple[int, int]], min_support: int = 2) -> list[int]:
    """Keep candidate line positions with enough independent Hough-segment
    support to be a real line rather than a stray digit-stroke artifact."""
    return sorted(pos for pos, support in clusters if support >= min_support)


def find_grid_line_positions(
    card_img: np.ndarray, edge_exclude_frac: float = 0.03, min_length_frac: float = 0.15
) -> tuple[list[int], list[int]]:
    """
    Detect the actual printed horizontal and vertical grid lines within
    a single card image, returning (horizontal_ys, vertical_xs) — the
    real divider positions, rather than assuming they're evenly spaced.

    Uses a probabilistic Hough transform (HoughLinesP) on Canny edges,
    rather than e.g. morphological opening, specifically because Hough
    tolerates GAPS in a line — important here since a real divider line
    gets broken up by the "FREE" cell's small serial-code sub-text, and
    sometimes by a digit's descender touching the line.

    Lines within edge_exclude_frac of either edge are discarded — those
    are the card's own outer border, not an internal divider; the outer
    boundary is handled separately via the implicit 0/total positions
    in _cell_boundaries. When more than GRID_SIZE - 1 candidates survive
    that, only the most strongly-supported ones are kept (see
    _select_best_lines) — real grid lines produce many overlapping Hough
    segments; spurious ones (e.g. a digit's horizontal stroke) produce
    only one or two.
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50,
        minLineLength=int(w * min_length_frac), maxLineGap=40,
    )

    horiz_ys, vert_xs = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5
            if abs(dy) <= 5 and length >= w * min_length_frac:
                horiz_ys.append((y1 + y2) // 2)
            elif abs(dx) <= 5 and length >= h * min_length_frac:
                vert_xs.append((x1 + x2) // 2)

    horiz_clusters = _cluster_1d(horiz_ys, tol=h * 0.02)
    vert_clusters = _cluster_1d(vert_xs, tol=w * 0.02)

    h_margin = h * edge_exclude_frac
    w_margin = w * edge_exclude_frac
    horiz_clusters = [(y, c) for y, c in horiz_clusters if h_margin < y < h - h_margin]
    vert_clusters = [(x, c) for x, c in vert_clusters if w_margin < x < w - w_margin]

    horiz_ys = _select_best_lines(horiz_clusters)
    vert_xs = _select_best_lines(vert_clusters)
    return horiz_ys, vert_xs


def _cell_boundaries(detected: list[int], total: int, tol_frac: float = 0.4) -> list[int]:
    """
    Turn detected internal divider positions into GRID_SIZE+1 cell
    boundaries (0, ..., total). Real printed grids are close to evenly
    spaced, so that's used as a strong prior: for each of the
    GRID_SIZE - 1 expected dividers, snap to the nearest detected line
    if one exists within tol_frac of a cell's width/height of the
    guess; otherwise keep the guess for that one divider.

    Each guess is the PREVIOUSLY CHOSEN boundary plus one equal step —
    not an independent i * equal_step from 0 — so that once a real
    divider has been found, the next guess is relative to it. This
    matters when two candidates both fall within tolerance: the one
    numerically closer to an independent guess isn't necessarily the
    one that keeps consistent spacing with what's already been chosen.

    NOTE: an earlier version of this function also tried snapping the
    two OUTER boundaries (0/total) to a nearby detected line, on the
    theory that hand-picked corners often leave slack between the crop
    edge and the card's real border. Tested against ground truth, that
    regressed accuracy (76% -> 71%, row 4 specifically got worse) —
    most likely because trim_grid_bottom's cut is a density-based
    heuristic, not necessarily anywhere near an actual printed line, so
    snapping "the nearest detected line" at the bottom often grabbed
    what was really the row 3/4 divider, corrupting row 4's crop rather
    than fixing it. Reverted. The row 0/row 4 edge-effect seen in real
    hall photos is real, but this wasn't the right fix for it.
    """
    equal_step = total / GRID_SIZE
    tol = equal_step * tol_frac

    boundaries = [0]
    for i in range(1, GRID_SIZE):
        guess = boundaries[-1] + equal_step
        candidates = [d for d in detected if abs(d - guess) <= tol]
        boundaries.append(min(candidates, key=lambda d: abs(d - guess)) if candidates else round(guess))
    boundaries.append(total)
    return boundaries


def segment_grid(warped: np.ndarray, margin_pct: float = 0.06, use_detected_lines: bool = True) -> list[list[np.ndarray]]:
    """
    Slice the flattened card into a GRID_SIZE x GRID_SIZE list of cell
    images. When use_detected_lines is True (the default), cell
    boundaries come from the actual printed grid lines (see
    find_grid_line_positions) rather than assuming even spacing —
    equal division is only ever an approximation, and small drift
    compounds across the card. margin_pct can be much smaller than it
    used to need to be now that boundaries are precise rather than
    approximate; it's still needed to avoid the divider lines
    themselves bleeding into the OCR crop.
    """
    height, width = warped.shape[:2]

    if use_detected_lines:
        horiz_ys, vert_xs = find_grid_line_positions(warped)
        row_bounds = _cell_boundaries(horiz_ys, height)
        col_bounds = _cell_boundaries(vert_xs, width)
    else:
        row_bounds = [round(i * height / GRID_SIZE) for i in range(GRID_SIZE + 1)]
        col_bounds = [round(i * width / GRID_SIZE) for i in range(GRID_SIZE + 1)]

    cells = []
    for row in range(GRID_SIZE):
        row_cells = []
        y0, y1 = row_bounds[row], row_bounds[row + 1]
        cell_h = y1 - y0
        margin_y = int(cell_h * margin_pct)
        for col in range(GRID_SIZE):
            x0, x1 = col_bounds[col], col_bounds[col + 1]
            cell_w = x1 - x0
            margin_x = int(cell_w * margin_pct)
            cell = warped[y0 + margin_y: y1 - margin_y, x0 + margin_x: x1 - margin_x]
            row_cells.append(cell)
        cells.append(row_cells)
    return cells


def preprocess_cell(cell: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Threshold a single cell to clean black-on-white text for OCR.
    method: 'otsu' (good for even lighting) or 'adaptive' (better if lighting
    varies across the card, e.g. shadow on one side).
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    if method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up small speckle noise left over from thresholding.
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Upscale small cells — Tesseract does better with more pixels per character.
    thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return thresh


def ocr_cell(cell: np.ndarray, psm: int = 7, method: str = "otsu") -> tuple[str, float]:
    """
    Run Tesseract on a single cell, restricted to digits.
    Returns (recognized_text, mean_confidence_0_to_100).
    """
    processed = preprocess_cell(cell, method=method)
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(
        processed, config=config, output_type=pytesseract.Output.DICT
    )
    words = [w for w in data["text"] if w.strip()]
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
    text = "".join(words).strip()
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf


def validate_cell(row: int, col: int, raw_text: str) -> dict:
    """Check an OCR'd cell value against the known valid range for its column."""
    if row == 2 and col == 2:
        return {"value": None, "is_free_space": True, "valid": True, "raw": raw_text}

    low, high = COLUMN_RANGES[col]
    is_valid = False
    value = None
    if raw_text.isdigit():
        value = int(raw_text)
        is_valid = low <= value <= high

    return {"value": value, "is_free_space": False, "valid": is_valid, "raw": raw_text}
