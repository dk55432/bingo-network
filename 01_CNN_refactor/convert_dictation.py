#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
COL_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]
COL_SIZES = [5, 5, 4, 5, 5]

def tok_int(tok):
    t = tok.strip().strip(",").strip()
    if t.upper() == "FREE":
        return None
    if t.isdigit():
        return int(t)
    v = WORDS.get(t.lower())
    if v is not None:
        return v
    raise ValueError(f"unrecognized token {t!r}")

def parse_sheet(sheet_lines):
    cards = {}
    cur = None
    for ln in sheet_lines:
        ln = ln.strip()
        if not ln or ln.startswith("#") and ln.startswith("# -"):
            continue
        m = re.match(r"[Cc]ards?\s*(one|two|three)\s*:", ln)
        if m:
            cur = {"one": 1, "two": 2, "three": 3}[m.group(1)]
            cards[cur] = []
            continue
        if cur is not None:
            cards[cur] += [tok_int(x) for x in ln.split(",")]
    return cards

def column_major_to_grid(items):
    assert len(items) == 24, f"expected 24 non-free cells, got {len(items)}"
    cols = []
    idx = 0
    for n in COL_SIZES:
        cols.append(items[idx:idx + n])
        idx += n
    grid = [[None] * 5 for _ in range(5)]
    for c, col in enumerate(cols):
        rows = [r for r in range(5) if not (r == 2 and c == 2)]
        for r, v in zip(rows, col):
            grid[r][c] = v
    grid[2][2] = "FREE"
    for c in range(5):
        lo, hi = COL_RANGES[c]
        for r in range(5):
            v = grid[r][c]
            if v == "FREE":
                assert (r, c) == (2, 2)
                continue
            assert lo <= v <= hi, f"col {c} value {v} outside {lo}-{hi}"
    return grid

def png_to_jpeg(src, dst):
    import cv2
    im = cv2.imread(str(src))
    assert im is not None, f"could not read {src}"
    assert cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95]), dst
    print(f"wrote {dst.name} ({dst.stat().st_size//1024}KB)")

ROOT = Path("/Users/davidkohn/Downloads/bingo-network")
SRC = ROOT / "01_CNN_refactor" / "manually_dictated_cards.txt"
GT = ROOT / "scan_card_numbers.txt"
PHONE = ROOT / "01_CNN_refactor" / "phone_sheets"
DUPS = {"in_1788419678.png", "in_1788456962.png"}

text = SRC.read_text().splitlines()
blocks = []
per_sheet = {}
cur_sheet = []
sheet_imgs = []
for ln in text:
    if re.match(r"in_\d+\.png\s*:$", ln.strip()):
        if cur_sheet:
            per_sheet[sheet_imgs[-1]] = cur_sheet
            cur_sheet = []
        sheet_imgs.append(ln.strip().rstrip(":"))
    else:
        cur_sheet.append(ln)
if cur_sheet:
    per_sheet[sheet_imgs[-1]] = cur_sheet
assert set(per_sheet) == {"in_1788419637.png", "in_1788436281.png"}, per_sheet.keys()

sheet_id = 31
for img, lines in per_sheet.items():
    cards = parse_sheet(lines)
    assert set(cards) == {1, 2, 3}, cards.keys()
    jpeg = PHONE / f"blue_sheet_{sheet_id:03d}.jpeg"
    png_to_jpeg(ROOT / "01_CNN_refactor" / img, jpeg)
    blk = [f"#blue_sheet_{sheet_id:03d}:"]
    for cid in (1, 2, 3):
        grid = column_major_to_grid(cards[cid])
        blk.append(f"#card{cid}")
        for row in grid:
            blk.append("[" + ", ".join(str(v) for v in row) + "]")
        blk.append("")
    blocks.append("\n".join(blk).rstrip() + "\n")
    print(f"\n== {img} -> {jpeg.name} ==\n" + "\n".join(blk))
    sheet_id += 1

shutil.copy2(GT, ROOT / "scan_card_numbers.txt.bak_dict2026-09-08")
with GT.open("a") as f:
    f.write("\n" + "\n".join(blocks))
print("\nappended to", GT.name)

for d in DUPS:
    p = ROOT / "01_CNN_refactor" / d
    p.unlink(missing_ok=True)
    print("deleted duplicate", d)

for img in per_sheet:
    p = ROOT / "01_CNN_refactor" / img
    p.unlink(missing_ok=True)
    print("removed source", img)