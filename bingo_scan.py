"""
Bingo card scanning pipeline.

Flow:
  1. Receive an uploaded photo of a bingo card.
  2. Find the card's rectangular outline and warp it to a flat, top-down view.
  3. Slice the flattened card into a 5x5 grid of cells.
  4. OCR each cell individually with Tesseract (digits only).
  5. Validate each recognized number against the known B-I-N-G-O column ranges.
  6. Return the grid as JSON for the client to review/correct before saving.

Requirements:
  pip install fastapi uvicorn python-multipart opencv-python-headless pytesseract numpy pillow

  You also need the Tesseract binary itself installed on the server, e.g.:
    Debian/Ubuntu: apt-get install -y tesseract-ocr
    macOS:         brew install tesseract
"""

import io
from typing import Optional

import cv2
import numpy as np
import pytesseract
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session

from database import create_db_and_tables, get_session
from models import BingoCard, Player

app = FastAPI()

# Adjust allow_origins to wherever player.html/scan.html are actually served from.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()

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


def load_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV BGR image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
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


def find_card_contour(img: np.ndarray) -> np.ndarray:
    """Locate the largest 4-sided contour in the image, assumed to be the card."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:5]:  # only bother checking the largest few
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    raise HTTPException(
        status_code=422,
        detail="Could not find a clear card outline. Retake the photo with better contrast/lighting.",
    )


def warp_card(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the card to a flat WARPED_SIDE x WARPED_SIDE square."""
    rect = order_points(corners.astype("float32"))
    dst = np.array(
        [[0, 0], [WARPED_SIDE - 1, 0], [WARPED_SIDE - 1, WARPED_SIDE - 1], [0, WARPED_SIDE - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (WARPED_SIDE, WARPED_SIDE))


def segment_grid(warped: np.ndarray) -> list[list[np.ndarray]]:
    """Slice the flattened card into a GRID_SIZE x GRID_SIZE list of cell images."""
    cell_side = WARPED_SIDE // GRID_SIZE
    cells = []
    for row in range(GRID_SIZE):
        row_cells = []
        for col in range(GRID_SIZE):
            y0, y1 = row * cell_side, (row + 1) * cell_side
            x0, x1 = col * cell_side, (col + 1) * cell_side
            # Shave a small margin off each cell to avoid grid-line noise at the edges.
            margin = int(cell_side * 0.12)
            cell = warped[y0 + margin: y1 - margin, x0 + margin: x1 - margin]
            row_cells.append(cell)
        cells.append(row_cells)
    return cells


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Threshold a single cell to clean black-on-white text for OCR."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # Otsu's threshold works well for consistent, well-lit printed text.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Upscale small cells — Tesseract does better with more pixels per character.
    thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return thresh


def ocr_cell(cell: np.ndarray) -> str:
    """Run Tesseract on a single cell, restricted to digits."""
    processed = preprocess_cell(cell)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed, config=config)
    return text.strip()


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


@app.post("/scan-card")
async def scan_card(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    file_bytes = await file.read()
    img = load_image(file_bytes)

    corners = find_card_contour(img)
    warped = warp_card(img, corners)
    cell_grid = segment_grid(warped)

    results = []
    needs_review = False
    for row in range(GRID_SIZE):
        result_row = []
        for col in range(GRID_SIZE):
            raw_text = ocr_cell(cell_grid[row][col])
            cell_result = validate_cell(row, col, raw_text)
            if not cell_result["valid"]:
                needs_review = True
            result_row.append(cell_result)
        results.append(result_row)

    return {"grid": results, "needs_review": needs_review}


class CreatePlayerRequest(BaseModel):
    player_display_name: str


@app.post("/players")
def create_player(payload: CreatePlayerRequest, session: Session = Depends(get_session)):
    # Skip this if players are already created elsewhere in your app —
    # just make sure the player_id you send to /cards already exists
    # (or points at your existing players table).
    player = Player(player_display_name=payload.player_display_name)
    session.add(player)
    session.commit()
    session.refresh(player)
    return player


class ConfirmCardRequest(BaseModel):
    # The client sends back the final grid after the user has reviewed/corrected
    # it — a plain 5x5 array of numbers (None for the free space), not the
    # richer per-cell objects /scan-card returns.
    player_id: str
    grid: list[list[Optional[int]]]


@app.post("/cards")
def confirm_card(payload: ConfirmCardRequest, session: Session = Depends(get_session)):
    if len(payload.grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in payload.grid):
        raise HTTPException(status_code=400, detail=f"Grid must be {GRID_SIZE}x{GRID_SIZE}")

    player = session.get(Player, payload.player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    card = BingoCard(player_id=payload.player_id, grid=payload.grid)
    session.add(card)
    session.commit()
    session.refresh(card)
    return card


@app.get("/cards/{card_id}")
def get_card(card_id: int, session: Session = Depends(get_session)):
    card = session.get(BingoCard, card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card


@app.get("/players/{player_id}")
def get_player(player_id: str, session: Session = Depends(get_session)):
    player = session.get(Player, player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player


@app.get("/players/{player_id}/cards")
def list_player_cards(player_id: str, session: Session = Depends(get_session)):
    player = session.get(Player, player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player.cards


# Serve scan.html, player.html, etc. as plain static files.
# This MUST be added after all the @app.get/@app.post routes above —
# it acts as a catch-all for any path not already matched, so if it were
# registered first it would shadow your API routes.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
