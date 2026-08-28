import os, sys, re
import cv2
import numpy as np

SRC = "/tmp/scan_cells_all"
OUT = "/tmp/scan_ds"
VALID_SHEETS = [26, 27, 28, 29, 30]

os.makedirs(OUT, exist_ok=True)


def clean(cell, thr=170, inset=6):
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bin = (cell < thr).astype(np.uint8)
    h, w = bin.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin, connectivity=8)
    if n <= 1:
        return cell
    for i in range(1, n):
        xs, ys, ws, hs, area = stats[i]
        if area < 30:
            continue
        touches = int(xs) <= inset or int(ys) <= inset or \
                  int(xs) + int(ws) >= w - inset or int(ys) + int(hs) >= h - inset
        # digit-size ink (touching a rule) must survive; all other touching
        # slivers are grid-rule remnants.
        digit_like = ws > 0.20 * w and hs > 0.45 * h
        if touches and not digit_like:
            bin[labels == i] = 0
    out = cell.copy()
    out[bin == 0] = 255
    return out


def main():
    if not os.path.isdir(SRC):
        print("run extract_scan_cells.py first")
        sys.exit(1)
    rows = []
    skipped = 0
    for fn in os.listdir(SRC):
        m = re.match(r"blue_sheet_0*(\d+)_card(\d)_r(\d)c(\d).*", fn)
        if not m:
            continue
        sheet, card, r, c = map(int, m.groups())
        if c == 2 and r == 2:
            skipped += 1
            continue
        img = cv2.imread(os.path.join(SRC, fn))
        if img is None:
            continue
        cl = clean(img)
        rows.append((sheet, card, r, c, cl))
    print(f"digit cells: {len(rows)} (excluded {skipped} FREE)")

    train = [x for x in rows if x[0] not in VALID_SHEETS]
    valid = [x for x in rows if x[0] in VALID_SHEETS]
    for split, items in (("train", train), ("valid", valid)):
        d = os.path.join(OUT, split)
        os.makedirs(d + "/cells", exist_ok=True)
        with open(os.path.join(OUT, f"{split}.csv"), "w") as f:
            f.write("sheet,card,r,c,path\n")
            for sheet, card, r, c, cl in items:
                name = f"blue_sheet_{sheet:03d}_card{card}_r{r}c{c}.png"
                cv2.imwrite(os.path.join(d, "cells", name), cl)
                f.write(f"{sheet},{card},{r},{c},cells/{name}\n")
        print(f"{split}: {len(items)}")


if __name__ == "__main__":
    main()