"""
Bingo card scanning: OCR pipeline + attaching the confirmed card to a
live player in a live game (no database — everything here is in-memory,
matching how Game/Player/BingoCard already work).

Flow:
  1. POST /scan-card: photo in, one OCR'd grid per detected card out
     (a strip can have multiple stacked cards — see split_into_cards()
     in pipeline.py). Stateless — doesn't touch game/player state, just
     runs the CV/OCR pipeline.
  2. Client reviews/corrects each grid (scan.html).
  3. POST /cards: {game_id, player_id, grids} in — one or more grids at
     once, since a single strip scan can produce several cards. Looks up
     the live Game and Player, converts each numeric grid to BingoCard's
     "B1"/"N31"/"FREE" label format, and calls player.add_card(...)
     directly on the live in-memory object — same object the websocket
     loop reads from.

Requirements:
  pip install fastapi python-multipart opencv-python-headless pytesseract numpy

  You also need the Tesseract binary itself installed, e.g.:
    Debian/Ubuntu: apt-get install -y tesseract-ocr
    macOS:         brew install tesseract
"""

import json
from typing import Optional

import numpy as np
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from bingo_card import BingoCard
from pipeline import (
    COLUMN_RANGES,
    GRID_SIZE,
    CardNotFoundError,
    find_card_contour,
    load_image,
    ocr_cell,
    segment_grid,
    split_into_cards,
    validate_cell,
    warp_card,
)

router = APIRouter()

COLUMN_LETTERS = ["B", "I", "N", "G", "O"]


def numeric_grid_to_labeled_grid(grid: list[list[Optional[int]]]) -> list[list[str]]:
    """
    Convert a plain 5x5 numeric grid (None for the free space) into
    BingoCard's label format, e.g. 12 in column 0 -> "B12", center -> "FREE".

    Raises ValueError if any non-free cell is missing a number or the
    number is outside its column's valid range — this is the last line
    of defense before a bad card enters live gameplay, so it's deliberately
    strict rather than silently accepting garbage.
    """
    labeled = []
    for row in range(GRID_SIZE):
        labeled_row = []
        for col in range(GRID_SIZE):
            if row == 2 and col == 2:
                labeled_row.append("FREE")
                continue

            value = grid[row][col]
            if value is None:
                raise ValueError(f"Missing number at row {row}, column {COLUMN_LETTERS[col]}")

            low, high = COLUMN_RANGES[col]
            if not (low <= value <= high):
                raise ValueError(
                    f"{value} is out of range for column {COLUMN_LETTERS[col]} "
                    f"({low}-{high}) at row {row}"
                )
            labeled_row.append(f"{COLUMN_LETTERS[col]}{value}")
        labeled.append(labeled_row)
    return labeled


def _ocr_one_card(card_img) -> dict:
    """Run the per-cell OCR loop over a single card's already-cropped
    (header-free) grid image, returning {"grid": ..., "needs_review": ...}."""
    results = []
    needs_review = False
    for row in range(GRID_SIZE):
        result_row = []
        for col in range(GRID_SIZE):
            raw_text, confidence = ocr_cell(card_img[row][col])
            cell_result = validate_cell(row, col, raw_text)
            cell_result["confidence"] = round(confidence, 1)
            if not cell_result["valid"]:
                needs_review = True
            result_row.append(cell_result)
        results.append(result_row)
    return {"grid": results, "needs_review": needs_review}


@router.post("/scan-card")
async def scan_card(file: UploadFile, corners: Optional[str] = Form(None)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    file_bytes = await file.read()
    try:
        img = load_image(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if corners:
        # Manually picked corners from scan.html's drag handles — trust
        # these completely and skip auto-detection. Expected shape:
        # [[x, y], [x, y], [x, y], [x, y]] in original-image pixel coords,
        # any corner order (warp_card/order_points sorts them out).
        try:
            pts = json.loads(corners)
            if not (isinstance(pts, list) and len(pts) == 4 and all(len(p) == 2 for p in pts)):
                raise ValueError("must be a list of exactly 4 [x, y] pairs")
            card_corners = np.array(pts, dtype="float32")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid corners: {e}")
    else:
        try:
            card_corners = find_card_contour(img)
        except CardNotFoundError as e:
            raise HTTPException(status_code=422, detail=str(e))

    warped = warp_card(img, card_corners)

    # A strip can have one card or several stacked cards — split_into_cards
    # finds each card's header band and returns one cropped grid image per
    # card (falling back to "the whole thing is one card" if no header
    # band is detected at all, e.g. an already-tightly-cropped single card).
    card_images = split_into_cards(warped)
    cards = [_ocr_one_card(segment_grid(card_img)) for card_img in card_images]

    return {"cards": cards}


@router.get("/games/{game_id}/players/{player_id}")
def get_live_player(game_id: str, player_id: str, request: Request):
    game_manager = request.app.state.game_manager
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    player = game.get_player(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found in this game")
    return {"player_id": player.player_id, "display_name": player.display_name}


class ConfirmCardsRequest(BaseModel):
    game_id: str
    player_id: str
    # One plain 5x5 array of numbers per card, after client-side
    # review/correction; None for each free space. A single strip scan
    # can produce several of these at once.
    grids: list[list[list[Optional[int]]]]


@router.post("/cards")
async def confirm_cards(payload: ConfirmCardsRequest, request: Request):
    # game_manager (and, for the websocket push below, the connection
    # manager) are stashed on app.state by main.py — see the setup note
    # below. Using app.state instead of importing main.py directly avoids
    # a circular import (main.py already imports this module).
    game_manager = request.app.state.game_manager

    game = game_manager.get_game(payload.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    player = game.get_player(payload.player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found in this game")

    if not payload.grids:
        raise HTTPException(status_code=400, detail="No grids to save")

    for i, grid in enumerate(payload.grids):
        if len(grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in grid):
            raise HTTPException(status_code=400, detail=f"Grid {i} must be {GRID_SIZE}x{GRID_SIZE}")

    try:
        labeled_grids = [numeric_grid_to_labeled_grid(g) for g in payload.grids]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    saved_cards = []
    for labeled_grid in labeled_grids:
        card = BingoCard(card_id=None, player_id=payload.player_id, grid=labeled_grid)
        player.add_card(card)
        saved_cards.append(card)
    game.touch()

    # If this player has a live websocket connection open elsewhere (e.g. a
    # player.html tab left open on another device/tab while they scan on
    # their phone), push the updated card list to it — mirrors
    # send_cards_to_player() in main.py. Best-effort: scanning still
    # succeeds even if this fails or there's no live connection.
    connection_manager = getattr(request.app.state, "connection_manager", None)
    if connection_manager is not None and player.websocket is not None:
        try:
            await connection_manager.send_to_player(
                player.websocket,
                json.dumps({
                    "type": "cards",
                    "cards": [c.to_dict() for c in player.cards],
                }),
            )
        except Exception:
            pass  # don't fail the scan just because the push failed

    return [c.to_dict() for c in saved_cards]
