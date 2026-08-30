"""Offline 3-way comparison on held-out phone cells.

Sheets 25-30 from BOTH photo locations (phone_sheets, phone_sheets3):
  1. CNN raw        : per-cell argmax over 75 (no constraints)
  2. CNN constrained: per-column linear-assignment decoding
  3. Tesseract      : pipeline.ocr_cell on the same BGR cell

Compares exact per-cell match against scan-dictation GT.
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cnn_reader
from evaluate_photo_sheets import load_photo, normalize
from evaluate_scan_cells import parse_truth
from photo_sheet_cells import sheet_to_cells_teal

HELD_OUT = range(25, 31)
BATCHES = [Path(__file__).parent / "phone_sheets",
           Path(__file__).parent / "phone_sheets3",
           Path(__file__).parent / "phone_sheets2"]


def truth_cell(truth, s, cid, r, c):
    gt = truth.get((s, cid))
    return gt[r][c] if gt else None


def main():
    from pipeline import ocr_cell
    truth = parse_truth()
    model = cnn_reader.load_model()

    stats = {"cnn_raw": {"ok": 0, "tot": 0},
             "cnn_con": {"ok": 0, "tot": 0},
             "cnn_mask": {"ok": 0, "tot": 0},
             "tess": {"ok": 0, "tot": 0},
             "tess_valid": 0}
    cnn_con_boards = {"exact24": 0, "tot": 0}
    col_ok = {c: {"ok": 0, "tot": 0} for c in range(5)}

    for batch in BATCHES:
        tag = batch.name
        # phone_sheets2 = single-card shots, labels unrecoverable -> skip
        if "sheets2" in tag:
            continue
        for path in sorted(batch.glob("*.jpeg")):
            s = int(re.search(r"(\d+)(?=\.jpe?g$)", path.name).group(1))
            if s not in HELD_OUT:
                continue
            img = normalize(load_photo(path))
            cells = sheet_to_cells_teal(img)
            by_card = {}
            for cid, r, c, cell in cells:
                by_card.setdefault(cid, {})[(r, c)] = cell
            for cid, ccells in sorted(by_card.items()):
                lgrid = {}
                n_grid = [[None] * 5 for _ in range(5)]
                t_grid = [[None] * 5 for _ in range(5)]
                for (r, c), cell in sorted(ccells.items()):
                    if (r, c) == (2, 2):
                        continue
                    target = truth_cell(truth, s, cid, r, c)
                    if target is None:
                        continue
                    stats["cnn_raw"]["tot"] += 1
                    stats["cnn_mask"]["tot"] += 1
                    stats["tess"]["tot"] += 1
                    lg = cnn_reader.cell_logits(model, cell)
                    lgrid[(r, c)] = lg

                    rn, rvk = cnn_reader.decode_cell_args(lg, c)
                    n_grid[r][c] = (rn, rvk)

                    lo, hi = cnn_reader.COLUMN_RANGES[c]
                    msub = np.array([lg[cnn_reader.NUM_TO_IDX[n]] for n in
                                     range(lo, hi + 1)])
                    mnum = lo + int(np.argmax(msub))
                    _ = mnum

                    text, conf = ocr_cell(cell)
                    tval = None
                    try:
                        tval = int("".join(ch for ch in text if ch.isdigit()))
                    except ValueError:
                        tval = None
                    t_grid[r][c] = tval
                    if tval is not None and lo <= tval <= hi:
                        stats["tess_valid"] += 1

                    if mnum == target:
                        stats["cnn_mask"]["ok"] += 1
                    if rn == target:
                        stats["cnn_raw"]["ok"] += 1
                    if tval == target:
                        stats["tess"]["ok"] += 1
                    if rvk:
                        col_ok[c]["tot"] += 1
                        if rn == target:
                            col_ok[c]["ok"] += 1

                numbers, _ = cnn_reader.decode_card(lgrid)
                nonexact = 0
                bad = []
                for (r, c), cell in sorted(ccells.items()):
                    if (r, c) == (2, 2):
                        continue
                    target = truth_cell(truth, s, cid, r, c)
                    if target is None:
                        continue
                    stats["cnn_con"]["tot"] += 1
                    if numbers[r][c] == target:
                        stats["cnn_con"]["ok"] += 1
                    else:
                        nonexact += 1
                        bad.append((r, c, target, numbers[r][c]))
                if s in range(28, 31) and batch is not BATCHES[0]:
                    pass
                cnn_con_boards["tot"] += 1
                if nonexact == 0:
                    cnn_con_boards["exact24"] += 1
                elif cnn_con_boards["tot"] < 40:
                    print(f"  [{tag}] s{s} c{cid}: {24-nonexact}/24 "
                          + ", ".join(f"r{r}c{c} {t}->{p}" for r, c, t, p in bad[:6]))

    def acc(k):
        a = stats[k]
        return f"{a['ok']}/{a['tot']} = {a['ok']/a['tot']:.4f}"

    print("\n=== held-out sheets 25-30, both locations ===")
    print("cnn raw        :", acc("cnn_raw"))
    print("cnn mask(col)  :", acc("cnn_mask"))
    print("cnn constrained:", acc("cnn_con"))
    print("tesseract      :", acc("tess"),
          f"[valid-in-range {stats['tess_valid']}/{stats['tess']['tot']}"
          f" = {stats['tess_valid']/stats['tess']['tot']:.3f}]")
    print("constrained boards exact 24/24:",
          f"{cnn_con_boards['exact24']}/{cnn_con_boards['tot']}")
    print("per-column cnn_raw (range-valid only) acc:",
          {c: f"{col_ok[c]['ok']}/{col_ok[c]['tot']}"
           for c in range(5)})


if __name__ == "__main__":
    main()