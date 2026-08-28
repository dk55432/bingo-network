"""Build a glyph-level digit training set from scan cells + ground truth.

Cell crops live in /tmp/scan_ds/{train,valid}/cells (from build_scan_dataset).
Labels come from scan_card_numbers.txt (5x5 row-major grids per sheet/card,
FREE at r2c2). Each cell is split into 1-2 glyph crops with
recognize_cards.split_digits; glyphs are labelled tens/ones from the truth
number by left-to-right order.
"""

import os
import re
import ast
import cv2
import numpy as np

DS = "/tmp/scan_ds"
OUT = "/tmp/scan_digits"
GT = "/Users/davidkohn/Downloads/bingo-network/scan_card_numbers.txt"


def parse_truth():
    truth = {}
    sheet = None
    card = None
    rows = []
    for raw in open(GT):
        ln = raw.strip()
        if not ln:
            if sheet is not None and card is not None and rows:
                truth[(sheet, card)] = rows
            rows = []
            continue
        if ln.startswith("#"):
            m = re.match(r"#blue_sheet_0*(\d+):", ln)
            if m:
                sheet = int(m.group(1))
                card = None
                rows = []
                continue
            m = re.match(r"#card(\d)", ln)
            if m:
                card = int(m.group(1))
                rows = []
                continue
            continue
        rows.append(ast.literal_eval(ln.replace("FREE", '"FREE"')))
    if sheet is not None and card is not None and rows:
        truth[(sheet, card)] = rows
    return truth


def tight(gray):
    bin = (gray < 170).astype(np.uint8)
    ys, xs = np.nonzero(bin)
    if len(ys) == 0:
        return gray
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin, 8)
    if n <= 1:
        return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    best = None
    best_area = 0
    for i in range(1, n):
        if stats[i][cv2.CC_STAT_AREA] > best_area:
            best_area = stats[i][cv2.CC_STAT_AREA]
            best = i
    ys, xs = np.nonzero(lab == best)
    if len(ys) == 0:
        return gray
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def split_glyphs(gray, ndigits):
    """Return ndigits glyph crops (1 or 2) from a clean cell.

    Uses the horizontal ink profile of the cell's bounding region: for a
    single digit take the whole bbox; for two digits split the bbox at
    the fewest-ink x position within the middle third, then tight-crop
    each half to its own ink (valley cut may clip glyph edges).
    """
    bin = (gray < 170).astype(np.uint8)
    ys, xs = np.nonzero(bin)
    if len(ys) == 0:
        return []
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if ndigits == 1:
        return [tight(gray[y0:y1 + 1, x0:x1 + 1])]
    region = bin[y0:y1 + 1, x0:x1 + 1]
    col_ink = region.sum(axis=0).astype(np.float32)
    m = len(col_ink)
    lo, hi = int(0.30 * m), int(0.70 * m)
    if hi - lo < 4:
        cut = (x0 + x1) // 2
    else:
        cut = x0 + lo + int(np.argmin(col_ink[lo:hi]))
    if cut - x0 < 4 or x1 - cut < 4:
        return []
    return [tight(gray[y0:y1 + 1, x0:cut]),
            tight(gray[y0:y1 + 1, cut:x1 + 1])]


def main():
    truth = parse_truth()
    for split in ("train", "valid"):
        d = os.path.join(OUT, split)
        os.makedirs(d, exist_ok=True)
        counts = {}
        saved = 0
        skipped = 0
        for fn in sorted(os.listdir(os.path.join(DS, split, "cells"))):
            m = re.match(r"blue_sheet_0*(\d+)_card(\d)_r(\d)c(\d)", fn)
            if not m:
                continue
            sheet, card, r, c = map(int, m.groups())
            grid = truth.get((sheet, card))
            if grid is None:
                skipped += 1
                continue
            num = grid[r][c]
            if num == "FREE":
                continue
            s = str(int(num))
            gray = cv2.imread(os.path.join(DS, split, "cells", fn),
                              cv2.IMREAD_GRAYSCALE)
            glyphs = split_glyphs(gray, len(s))
            labels = [int(d) for d in s]
            if len(glyphs) != len(labels):
                skipped += 1
                continue
            for gi, (g, lab) in enumerate(zip(glyphs, labels)):
                if g is None or g.size == 0:
                    skipped += 1
                    continue
                name = f"l{lab}_s{sheet:03d}c{card}r{r}c{c}g{gi}.png"
                cv2.imwrite(os.path.join(d, name), g)
                counts[lab] = counts.get(lab, 0) + 1
                saved += 1
        print(f"{split}: saved={saved} skipped={skipped}")
        print(f"  per-class: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()