"""
Standalone debug tool for the bingo scanning pipeline.

Feed it a photo, get back a folder of intermediate images (edge detection,
detected contour, warped card, per-cell OCR results) so you can see exactly
where the pipeline is going wrong — no phone, no server, no app in the way.

Usage:
    python debug_scan.py path/to/card.jpg
    python debug_scan.py path/to/card.jpg --psm 8 --threshold adaptive

Output goes to debug_output/<image_name>/:
    01_edges.jpg                     - what Canny edge detection saw
    02_contour.jpg                   - the outline it decided on (red box) -
                                        for the WHOLE strip if there are
                                        multiple stacked cards, not one card
    03_warped.jpg                    - the flattened, top-down strip
    card_{n}/card.jpg                - that card's cropped grid (header
                                        and footer/gap already excluded)
    card_{n}/grid_overlay.jpg        - that card with each cell's OCR
                                        result drawn on top (green = valid,
                                        red = flagged)
    card_{n}/cells/row{r}_col{c}.jpg - each individual preprocessed cell

One card_{n}/ subdirectory per detected card — a photo of a single card
produces card_0/ only; a 3-card strip produces card_0/, card_1/, card_2/.

Also prints a text summary of every cell's raw OCR text and confidence,
which is often the fastest way to spot a systematic problem (e.g. every
cell in one column is empty, or confidence is uniformly low).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

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

COLUMN_LETTERS = ["B", "I", "N", "G", "O"]


def run(image_path: Path, psm: int, threshold_method: str, margin_pct: float, corners_arg: str | None = None):
    out_dir = Path("debug_output") / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = image_path.read_bytes()
    try:
        img = load_image(file_bytes)
    except ValueError as e:
        print(f"FAILED to load image: {e}")
        sys.exit(1)

    if corners_arg:
        # Manual corners, same as scan.html's drag handles send to the
        # real app — skips auto-detection entirely, same as bingo_scan.py
        # does when the client provides them.
        try:
            pts = json.loads(corners_arg)
            if not (isinstance(pts, list) and len(pts) == 4 and all(len(p) == 2 for p in pts)):
                raise ValueError("must be a list of exactly 4 [x, y] pairs")
            corners = np.array(pts, dtype="float32")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"FAILED to parse --corners: {e}")
            sys.exit(1)
        print("Using manually-provided corners (auto-detection skipped)")
    else:
        debug = {}
        try:
            corners = find_card_contour(img, debug=debug)
        except CardNotFoundError as e:
            # Still save what we found so you can see WHY it failed.
            if "edges" in debug:
                cv2.imwrite(str(out_dir / "01_edges.jpg"), debug["edges"])
            if "contour_overlay" in debug:
                cv2.imwrite(str(out_dir / "02_contour.jpg"), debug["contour_overlay"])
            print(f"FAILED to find card outline: {e}")
            print(f"Check {out_dir}/01_edges.jpg — if the card's edges aren't clean")
            print("continuous lines there, that's your problem (lighting/background/contrast).")
            print()
            print("If you know this photo needs manual corners in the real app too, skip")
            print("auto-detection here the same way: python debug_scan.py "
                  f"{image_path} --corners '[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]'")
            sys.exit(1)

        cv2.imwrite(str(out_dir / "01_edges.jpg"), debug["edges"])
        cv2.imwrite(str(out_dir / "02_contour.jpg"), debug["contour_overlay"])
        print(f"Card outline found OK -> {out_dir}/02_contour.jpg (check the red box is tight on the card)")

    warped = warp_card(img, corners)
    cv2.imwrite(str(out_dir / "03_warped.jpg"), warped)
    print(f"Warped card -> {out_dir}/03_warped.jpg (check this looks flat/square, not skewed)")

    card_images = split_into_cards(warped)
    print(f"Detected {len(card_images)} card(s) on this strip")
    print()

    for card_idx, card_img in enumerate(card_images):
        card_dir = out_dir / f"card_{card_idx}"
        (card_dir / "cells").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(card_dir / "card.jpg"), card_img)

        cell_grid = segment_grid(card_img, margin_pct=margin_pct)
        overlay = card_img.copy()
        cell_h = card_img.shape[0] // GRID_SIZE
        cell_w = card_img.shape[1] // GRID_SIZE

        print(f"=== Card {card_idx} ===")
        print(f"{'cell':<8}{'raw text':<10}{'value':<8}{'valid?':<8}{'confidence'}")
        print("-" * 50)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell_img = cell_grid[row][col]
                cv2.imwrite(str(card_dir / "cells" / f"row{row}_col{col}.jpg"), cell_img)

                if row == 2 and col == 2:
                    print(f"{COLUMN_LETTERS[col]}{row:<7}{'FREE':<10}{'-':<8}{'-':<8}-")
                    continue

                raw_text, confidence = ocr_cell(cell_img, psm=psm, method=threshold_method)
                result = validate_cell(row, col, raw_text)
                label = f"{COLUMN_LETTERS[col]}{row}"
                print(
                    f"{label:<8}{raw_text or '(empty)':<10}"
                    f"{str(result['value']):<8}{str(result['valid']):<8}{confidence:.0f}"
                )

                # Draw the result onto the overlay for a quick visual scan.
                x = col * cell_w + 8
                y = row * cell_h + cell_h // 2
                color = (0, 160, 0) if result["valid"] else (0, 0, 220)
                text = raw_text if raw_text else "?"
                cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite(str(card_dir / "grid_overlay.jpg"), overlay)
        print(f"Card {card_idx} overlay -> {card_dir}/grid_overlay.jpg")
        print(f"Card {card_idx} cells -> {card_dir}/cells/")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Path to a card photo (.jpg/.png)")
    parser.add_argument(
        "--corners", type=str, default=None,
        help="Manually specify the 4 card corners as a JSON list, e.g. "
        "'[[120,80],[900,95],[890,1200],[110,1180]]', in original-image pixel "
        "coordinates, any order. Skips auto-detection entirely — use this for "
        "any photo where auto-detection fails or picks the wrong outline, "
        "same as scan.html's drag-corner picker does for the real app.",
    )
    parser.add_argument("--psm", type=int, default=7, help="Tesseract page segmentation mode (try 7 or 8)")
    parser.add_argument(
        "--threshold", dest="threshold_method", choices=["otsu", "adaptive"], default="otsu",
        help="Cell thresholding method: 'otsu' (even lighting) or 'adaptive' (uneven lighting/shadows)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.06,
        help="Fraction of each cell to trim as margin (increase if grid lines are getting OCR'd, "
        "decrease if digits are getting cut off). Matches pipeline.py's segment_grid default.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"No such file: {args.image}")
        sys.exit(1)

    run(
        args.image, psm=args.psm, threshold_method=args.threshold_method,
        margin_pct=args.margin, corners_arg=args.corners,
    )
