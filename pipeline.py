"""
Pure image-processing + OCR pipeline for bingo cards. No FastAPI dependency
here on purpose — this module gets imported both by the running API
(bingo_scan.py) and by the standalone debug_scan.py CLI tool.
"""

import cv2
import numpy as np
import pytesseract

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
    """Perspective-warp the card to a flat WARPED_SIDE x WARPED_SIDE square."""
    rect = order_points(corners.astype("float32"))
    dst = np.array(
        [[0, 0], [WARPED_SIDE - 1, 0], [WARPED_SIDE - 1, WARPED_SIDE - 1], [0, WARPED_SIDE - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (WARPED_SIDE, WARPED_SIDE))


def segment_grid(warped: np.ndarray, margin_pct: float = 0.12) -> list[list[np.ndarray]]:
    """Slice the flattened card into a GRID_SIZE x GRID_SIZE list of cell images."""
    cell_side = WARPED_SIDE // GRID_SIZE
    cells = []
    for row in range(GRID_SIZE):
        row_cells = []
        for col in range(GRID_SIZE):
            y0, y1 = row * cell_side, (row + 1) * cell_side
            x0, x1 = col * cell_side, (col + 1) * cell_side
            margin = int(cell_side * margin_pct)
            cell = warped[y0 + margin: y1 - margin, x0 + margin: x1 - margin]
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
