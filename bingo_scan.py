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

  Optional CNN reader (READER=cnn): additionally needs torch, torchvision,
  scipy, pillow. The CNN reader lives in 01_CNN_refactor/cnn_reader.py and
  uses the whole-cell model cell_classifier_phone.pth (column-constrained
  decode, ~64% cell-accurate on held-out photos vs ~23% for Tesseract).

  You also need the Tesseract binary itself installed, e.g.:
    Debian/Ubuntu: apt-get install -y tesseract-ocr
    macOS:         brew install tesseract
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
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

logger = logging.getLogger(__name__)

COLUMN_LETTERS = ["B", "I", "N", "G", "O"]

# Reviewed/corrected cells land here as labeled training images, one per
# number (durable — this is the "memory" the CNN retrains on).
LEARNING_DIR = Path(__file__).parent / "01_CNN_refactor" / "learning_cells"


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


# The CNN reader lives in 01_CNN_refactor and needs torch; it is imported
# lazily (and its model is cached) so the default Tesseract path stays
# free of the heavy dependencies.
_CNN_MODEL = None
_CNN_RDIR = None


def _cnn_module():
    global _CNN_RDIR
    rdir = Path(__file__).parent / "01_CNN_refactor"
    if str(rdir) not in sys.path:
        sys.path.insert(0, str(rdir))
    _CNN_RDIR = rdir
    import cnn_reader  # noqa: PLC0415
    return cnn_reader


def _cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is None:
        _CNN_MODEL = _cnn_module().load_model()
    return _CNN_MODEL


@router.get("/scan-debug/{ts}.png")
async def scan_debug_overlay(ts: int):
    """Serve a scan's alignment overlay (the reader draws a green box per
    detected cell on the normalized photo) so the client can verify the
    cells line up with the physical sheet before saving.  Keyed by the
    integer scan id — no path traversal possible."""
    f = Path("/tmp/cnn_reader_debug") / f"overlay_{ts}.png"
    if not f.is_file():
        raise HTTPException(status_code=404, detail="no such scan overlay")
    return FileResponse(f, media_type="image/png")


@router.get("/scan-debug-in/{ts}.png")
async def scan_debug_in(ts: int):
    """Serve a scan's raw normalized input photo (no overlay) — used as the
    tap-target image for the assisted scan."""
    f = Path("/tmp/cnn_reader_debug") / f"in_{ts}.png"
    if not f.is_file():
        raise HTTPException(status_code=404, detail="no such scan photo")
    return FileResponse(f, media_type="image/png")


@router.post("/scan-card")
async def scan_card(file: UploadFile, corners: Optional[str] = Form(None)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    file_bytes = await file.read()

    # Reader selection: READER=cnn uses the whole-cell CNN reader on the
    # full photo (EXIF-correct + teal-band geometry + column-constrained
    # decode).  READER=homography uses SIFT+homography header detection
    # with per-card column projection.  READER=tesseract (default) keeps
    # the old contour/corner -> Hough grid -> Tesseract path.
    reader = os.environ.get("READER", "tesseract")
    if reader == "cnn":
        cr = _cnn_module()
        result = cr.read_sheet_bytes(_cnn_model(), file_bytes)
        if result.get("error"):
            raise HTTPException(
                status_code=422,
                detail={"error": result["error"], "debug": result.get("debug", {})})
        return {"cards": result["cards"], "scan_id": result.get("scan_id"),
            "debug": result.get("debug", {})}
    elif reader == "homography":
        # Import homography pipeline lazily (in 01_CNN_refactor/)
        sys.path.insert(0, str(Path(__file__).parent / "01_CNN_refactor"))
        from pipeline_homography import read_sheet_homography
        from cnn_reader import load_model
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        results, ts = read_sheet_homography(img)
        # Convert to same format as CNN reader
        cards = []
        for res in results:
            cards.append({
                "grid": res["grid"],
                "needs_review": False  # TODO: add confidence check
            })
        dump_in = f"/tmp/cnn_reader_debug/in_{ts}.png"
        dump_overlay = f"/tmp/cnn_reader_debug/overlay_{ts}.png"
        return {"cards": cards, "scan_id": str(ts), "debug": {"dump_in": dump_in, "dump_overlay": dump_overlay, "ts": str(ts), "partial_sheet": len(results) < 3, "warped": False}}

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


@router.post("/scan-assist")
async def scan_assist(file: UploadFile, band_tops: str = Form(...)):
    """Assisted scan for washed-out sheets (e.g. gray cards) when the auto
    header-color detection can't find the card bands: the client taps the
    TOP of each of the 3 gray header bars on the normalized photo
    (/scan-debug-in/{scan_id}.png) and sends those pixel rows here.  The
    card grid is then pinned to those taps before running the same
    whole-cell CNN decode."""
    if os.environ.get("READER", "tesseract") != "cnn":
        raise HTTPException(status_code=400,
                            detail="Assisted scan needs READER=cnn")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="Uploaded file must be an image")
    try:
        tops = json.loads(band_tops)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400,
                            detail="band_tops must be a JSON array of 3 rows")
    if not (isinstance(tops, list) and len(tops) == 3
            and all(isinstance(v, (int, float)) and 0 <= v <= 65535
                    for v in tops)):
        raise HTTPException(
            status_code=400,
            detail="band_tops must be exactly 3 pixel rows [y1, y2, y3]")

    file_bytes = await file.read()
    try:
        # Re-normalize the ORIGINAL photo (deterministic) so tap coordinates
        # from the /scan-debug-in dump match the image actually processed.
        result = _cnn_module().read_sheet_bytes(
            _cnn_model(), file_bytes, forced_bands=[float(t) for t in tops])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result.get("error"):
        raise HTTPException(
            status_code=422,
            detail={"error": result["error"],
                    "debug": result.get("debug", {})})
    return {"cards": result["cards"], "scan_id": result.get("scan_id"),
        "debug": result.get("debug", {})}


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
    # Echoed back from /scan-card so the reader's saved cell crops (see
    # cnn_reader._persist_pending_cells) can be paired with these
    # confirmed grids to grow the CNN training data.
    scan_id: Optional[str] = None


def _save_learning_cells(payload: ConfirmCardsRequest) -> int:
    """Pair the reader's saved cell crops (cnn_reader._persist_pending_cells,
    keyed by scan_id) with the user's reviewed/corrected grids and write one
    labeled training image per cell into LEARNING_DIR/<number>/.

    Each saved cell's filename embeds the scan epoch AND the git SHA of the
    reader code that produced the crop (self-auditing corpus), and a
    correction tag "_x" when the user's final number differs from what the
    reader auto-recognized — so retraining can focus on cells the user
    actually corrected rather than the ones that sailed through.

    Fully best-effort: a missing scan_id, a stale/missing pending dir, or a
    pairing hiccup never fails the card save — the learning store just gets
    nothing that round."""
    if not payload.scan_id:
        return 0
    safe = "".join(ch for ch in str(payload.scan_id) if ch.isdigit())
    if not safe:
        return 0
    pending = Path("/tmp/phone_learning_pending") / safe
    if not pending.is_dir():
        logger.info("learning: no pending cells for scan %s", safe)
        return 0

    try:
        import learning_audit as la
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent
                               / "01_CNN_refactor"))
        import learning_audit as la
    cell_was_corrected = la.cell_was_corrected
    load_auto_json = la.load_auto_json
    repo_short_sha = la.repo_short_sha
    save_cell_name = la.save_cell_name

    sha = repo_short_sha(Path(__file__).resolve().parent)
    auto = load_auto_json(pending / "auto.json")
    n = 0
    corrected = 0
    try:
        for i, grid in enumerate(payload.grids):
            cid = i + 1  # reader names pending crops c1..cN in card order
            if auto and i < len(auto):
                auto_grid = auto[i]
            else:
                auto_grid = None
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if row == 2 and col == 2:
                        continue
                    val = grid[row][col]
                    if val is None:
                        continue
                    src = pending / f"c{cid}_r{row}c{col}.jpg"
                    if not src.is_file():
                        continue
                    auto_val = None
                    if auto_grid is not None:
                        try:
                            auto_val = auto_grid[row][col]
                        except (IndexError, TypeError):
                            auto_val = None
                    was_corrected = cell_was_corrected(auto_val, val)
                    out_dir = LEARNING_DIR / str(val)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dst = out_dir / save_cell_name(
                        safe, sha, cid, row, col, was_corrected)
                    if dst.exists():
                        continue
                    shutil.copyfile(src, dst)
                    n += 1
                    if was_corrected:
                        corrected += 1
        shutil.rmtree(pending, ignore_errors=True)
    except Exception:
        logger.exception("learning: failed pairing crops for scan %s", safe)
    if n:
        logger.info("learning: captured %d confirmed cells from scan %s "
                    "(%d corrected)", n, safe, corrected)
    return n


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

    # Learning loop: only after the cards are saved successfully, pair the
    # scanned cell crops with the confirmed grids and grow the training set.
    _save_learning_cells(payload)

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


class RejectScanRequest(BaseModel):
    scan_id: Optional[str] = None


@router.post("/scan/reject")
async def reject_scan(payload: RejectScanRequest):
    """User discarded a scan review (Cancel — discard this scan in scan.html).
    Purges that scan's saved crops and debug dumps so an un-accepted read
    never lingers in phone_learning_pending and can't pollute retraining.
    Accepted scans never reach here: /cards copies them into learning_cells
    and _save_learning_cells already deletes the pending dir itself. The
    rest (scans whose reviews were abandoned) can be purged in bulk with
    01_CNN_refactor/clean_scan_artifacts.py."""
    if not payload.scan_id:
        return {"deleted": False}
    safe = "".join(ch for ch in str(payload.scan_id) if ch.isdigit())
    if not safe:
        return {"deleted": False}
    pending = Path("/tmp/phone_learning_pending") / safe
    deleted = False
    if pending.is_dir():
        shutil.rmtree(pending, ignore_errors=True)
        logger.info("learning: discarded scan %s (user rejected)", safe)
        deleted = True
    dbg = Path("/tmp/cnn_reader_debug")
    for name in (f"in_{safe}.png", f"overlay_{safe}.png", f"orig_{safe}.jpg"):
        (dbg / name).unlink(missing_ok=True)
    return {"deleted": deleted}
