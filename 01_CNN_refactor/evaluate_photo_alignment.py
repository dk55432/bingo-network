"""Font-agnostic alignment check: phone cells vs the same sheet's scan cells.

The photo sheets are the SAME 30 printed sheets that were flatbed-scanned.
If photo geometry is correct, each phone cell shows the same digit in the
same place as the scan cell, so binary ink overlap is high regardless of
rendering. Geometry errors (shifted row/col) collapse IoU systematically
across a row or column; a single scattered mismatch is capture noise.

High mean IoU per cell with no systematic row/col failure => geometry good,
and the low accuracy through the scan-trained reader is purely a font
mismatch (expected, and why a phone classifier is being trained).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_photo_sheets import load_photo, normalize  # noqa: E402
from extract_scan_cells import sheet_to_cells  # noqa: E402
from photo_sheet_cells import sheet_to_cells_teal  # noqa: E402

SCANS = Path("/Users/davidkohn/Downloads/bingo-network/scans")
PHONE = Path(__file__).parent / "phone_sheets"
TARGET = 24


def ink_mask(cell):
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return (g < 150).astype(np.uint8)


def to_binary(mask):
    out = np.zeros((TARGET, TARGET), np.uint8)
    if mask.sum() == 0:
        return out, False
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ch, cw = y1 - y0 + 1, x1 - x0 + 1
    scale = min((TARGET - 6) / ch, (TARGET - 6) / cw)
    m = cv2.resize(mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8),
                   (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                   interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8)
    oy = (TARGET - m.shape[0]) // 2
    ox = (TARGET - m.shape[1]) // 2
    out[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
    return out, True


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / max(1, union)


def main():
    targets = sys.argv[1:] or sorted(Path(PHONE).glob("*.jpeg"))
    totals = []
    for path in sorted(targets):
        p = Path(path)
        m = __import__("re").search(r"(\d+)(?=\.jpe?g$)", p.name)
        if not m:
            continue
        s = int(m.group(1))
        scan_p = SCANS / f"blue_sheet_{s:03d}.png"
        if not scan_p.exists():
            print(f"{p.name}: no matching scan {scan_p.name}, skip")
            continue
        ph = sheet_to_cells_teal(normalize(load_photo(p)), return_geometry=True)
        sc = sheet_to_cells(cv2.imread(str(scan_p)), return_geometry=True)
        if not ph:
            print(f"{p.name}: no photo cells")
            continue
        phd = {(cid, r, c): cell for cid, r, c, cell, _ in ph}
        scd = {(cid, r, c): cell for cid, r, c, cell, _ in sc}
        print(f"== {p.name}: photo {len(phd)} cells, scan {len(scd)} cells")
        for cid in sorted({k[0] for k in phd}):
            n = ok = 0
            worst = []
            for r in range(5):
                for c in range(5):
                    if r == 2 and c == 2:
                        continue
                    key = (cid, r, c)
                    if key not in phd or key not in scd:
                        continue
                    pa, has_p = to_binary(ink_mask(phd[key]))
                    sa, has_s = to_binary(ink_mask(scd[key]))
                    n += 1
                    if has_p != has_s:
                        worst.append((r, c, 0.0, "ink?"))
                        continue
                    v = iou(pa, sa)
                    if v >= 0.5:
                        ok += 1
                    if v < 0.5:
                        worst.append((r, c, round(v, 2), ""))
            print(f"  card{cid}: {ok}/{n} cells IoU>=0.5"
                  + (f"; low: {worst}" if worst else ""))
        print(f"  photo-only cards: {sorted({k[0] for k in scd} - {k[0] for k in phd})}")


if __name__ == "__main__":
    main()