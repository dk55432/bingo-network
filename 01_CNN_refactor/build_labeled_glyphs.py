"""Build a phone-domain glyph dataset from photographed sheets, using the
scan ground truth as labels (the photos are of the SAME printed sheets).

Each labeled cell is segmented with the inference-time split_digits, so the
dataset representation matches exactly what the server reader will see.
Glyph-count vs digit-count mismatches are flagged: systematic mismatches
indicate geometry problems (a splitter/fuser or a blank cell).

Output: /tmp/phone_labeled_glyphs/{train,valid}/{0..9}/*.jpg  (28x28 canvas
via _render_glyph, same representation train_phone_digits.py consumes).
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_photo_sheets import load_photo, normalize  # noqa: E402
from evaluate_scan_cells import parse_truth  # noqa: E402
from photo_sheet_cells import sheet_to_cells_teal  # noqa: E402
from recognize_cards import _render_glyph, split_digits  # noqa: E402

PHONE = Path(__file__).parent / "phone_sheets"
DST = Path("/tmp/phone_labeled_glyphs")
FREE = (2, 2)


def main():
    truth = parse_truth()
    # per-sheet split: train on sheets 1-2, hold out sheet 3
    splits = {"train": (1, 2), "valid": (3,)}
    stats = {k: {"ok": 0, "mismatch": 0, "skip": 0} for k in splits}
    for path in sorted(PHONE.glob("*.jpeg")):
        import re
        m = re.search(r"(\d+)(?=\.jpe?g$)", path.name)
        if not m:
            continue
        s = int(m.group(1))
        split = next((k for k, nums in splits.items() if s in nums), None)
        if split is None:
            continue
        cells = sheet_to_cells_teal(normalize(load_photo(path)))
        by_card = {}
        for cid, r, c, cell in cells:
            by_card.setdefault(cid, {})[(r, c)] = cell
        print(f"== {path.name}: sheet {s} -> {split}, cards {sorted(by_card)}")
        for cid in sorted(by_card):
            gt = truth.get((s, cid))
            if gt is None:
                print(f"  card{cid}: no truth")
                continue
            for r in range(5):
                for c in range(5):
                    if (r, c) == FREE:
                        continue
                    cell = by_card[cid].get((r, c))
                    if cell is None:
                        stats[split]["skip"] += 1
                        continue
                    label = str(gt[r][c])
                    glyphs = split_digits(cell)
                    if len(glyphs) != len(label):
                        stats[split]["mismatch"] += 1
                        if len(glyphs) == 1 and len(label) == 1:
                            stats[split]["ok"] += 1
                            glyphs = glyphs
                        else:
                            print(f"  card{cid} r{r}c{c}: {len(glyphs)} "
                                  f"glyphs vs {len(label)} digits")
                            continue
                    for k, gl in enumerate(glyphs):
                        canvas = _render_glyph(gl)
                        if canvas is None:
                            stats[split]["skip"] += 1
                            continue
                        cls = label[k]
                        d = DST / split / cls
                        d.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(
                            str(d / f"sheet{s}_c{cid}_r{r}c{c}_{k}.jpg"),
                            canvas)
                        stats[split]["ok"] += 1
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()