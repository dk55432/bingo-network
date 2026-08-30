"""Three-way cell reader comparison on the phone-photo ground truth.

Ground truth: /Users/davidkohn/Downloads/bingo-network/manual_card_numbers.txt
(layout: #IMG_XXXX: header, then #card1..#card3 blocks of five rows).

For each GT photo:
  extract_card_grids -> leveled card crops (top->bottom), zip with the
  GT card blocks; report any count mismatch.
  extract_grid_cells -> 5x5 BGR cells per card.

Readers on identical cells:
  cnn_glyph  -> split_digits + classify_digit (with bingo col-range guard)
  cnn_raw    -> _predict_from_raw (whole-cell resize; the old 34% path)
  tesseract  -> pipeline.ocr_cell/validate_cell (the current /scan-card path)

The center FREE cell is exempt for all three. A card whose grid fit
fails counts as a miss for every reader on every cell.
"""

import ast
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict_digit import DigitClassifier  # noqa: E402
from parse_bingo_sheet import extract_card_grids, extract_grid_cells  # noqa: E402
from pipeline import ocr_cell, validate_cell  # noqa: E402
from recognize_cards import classify_digit, _predict_from_raw, split_digits  # noqa: E402

GT = "/Users/davidkohn/Downloads/bingo-network/manual_card_numbers.txt"
PHOTOS = "/Users/davidkohn/Downloads/bingo-network/01_CNN_refactor/sample_bingo_cards"
PHONE_MODEL = Path(__file__).parent / "digit_classifier_phone.pth"

COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]


def parse_truth():
    truth = {}
    img = card = None
    block = []
    for raw in open(GT):
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            m = re.match(r"#IMG_(\w+):", ln)
            if m:
                img = m.group(1)
                card = None
                block = []
                continue
            m = re.match(r"#card(\d)", ln)
            if m:
                if card is not None and img is not None and block:
                    truth[(img, card)] = ast.literal_eval(
                        "".join(block).rstrip(",").replace("FREE", '"FREE"'))
                card = int(m.group(1))
                block = []
                continue
            continue
        block.append(ln)
    if card is not None and img is not None and block:
        truth[(img, card)] = ast.literal_eval(
            "".join(block).rstrip(",").replace("FREE", '"FREE"'))
    return truth


def read_cell_glyphs(model, cell, col_range):
    """split_digits + classify_digit with the scan-style col-range guard."""
    glyphs = split_digits(cell)
    if not glyphs:
        return None, 0.0
    digs = [classify_digit(model, g) for g in glyphs]
    if len(glyphs) == 1:
        return digs[0][0], digs[0][1]
    lo, hi = col_range
    two = 10 * digs[0][0] + digs[1][0]
    conf2 = (digs[0][1] + digs[1][1]) / 2
    conf1 = digs[0][1]
    in2 = lo <= two <= hi
    in1 = lo <= digs[0][0] <= hi
    if in2 and (not in1 or conf2 >= conf1 + 1e-6):
        return two, conf2
    if in1 and (not in2 or conf1 >= conf2 + 1e-6):
        return digs[0][0], conf1
    return (two if in2 else None), max(conf1, conf2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(PHONE_MODEL, map_location=device))
    model.eval()

    truth = parse_truth()
    print(f"GT cards: {len(truth)}")

    # group GT by photo
    by_photo = {}
    for (img, card), rows in truth.items():
        by_photo.setdefault(img, {})[card] = rows

    stats = {"cnn_glyph": [0, 0], "cnn_raw": [0, 0], "tesseract": [0, 0]}
    full_cards = {"cnn_glyph": 0, "cnn_raw": 0, "tesseract": 0}
    n_cards = 0
    mismatches = []

    for img in sorted(by_photo):
        gt_blocks = by_photo[img]
        path = Path(PHOTOS) / f"IMG_{img}.jpeg"
        image = cv2.imread(str(path))
        if image is None:
            print(f"{img}: unreadable")
            continue
        cards, _, _ = extract_card_grids(image)
        print(f"{img}: GT {len(gt_blocks)} card(s), detected {len(cards)}")
        for i in range(1, len(gt_blocks) + 1):
            if i > len(cards):
                mismatches.append(f"{img} card{i}: no card detected")
                for key in stats:
                    full_cards[key] += 0
                continue
            res = extract_grid_cells(cards[i - 1])
            cells = res[0]
            fitted = res[4]
            truth_rows = gt_blocks[i]
            n_cards += 1
            per = {key: [0, 0] for key in stats}
            for r in range(5):
                for c in range(5):
                    if r == 2 and c == 2:
                        continue
                    target = truth_rows[r][c]
                    stats["cnn_glyph"][1] += 1
                    stats["cnn_raw"][1] += 1
                    stats["tesseract"][1] += 1
                    per["cnn_glyph"][1] += 1
                    per["cnn_raw"][1] += 1
                    per["tesseract"][1] += 1
                    if not fitted:
                        continue
                    cell = cells[r][c]
                    num, _ = read_cell_glyphs(model, cell, COLUMN_RANGES[c])
                    raw, _, _ = _predict_from_raw(
                        model, cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY))
                    text, _ = ocr_cell(cell)
                    v = validate_cell(r, c, text)
                    for key, pred in (("cnn_glyph", num), ("cnn_raw", raw),
                                      ("tesseract", v["value"])):
                        if pred == target:
                            stats[key][0] += 1
                            per[key][0] += 1
            for key in stats:
                if per[key][0] == per[key][1]:
                    full_cards[key] += 1

    print("\n===== CELL ACCURACY (excl FREE) =====")
    for key, (corr, tot) in stats.items():
        print(f"{key:>10}: {corr}/{tot} = {corr / tot if tot else 0:.4f} "
              f"(cards fully correct: {full_cards[key]}/{n_cards})")
    for m in mismatches:
        print("MISS:", m)


if __name__ == "__main__":
    import torch
    main()