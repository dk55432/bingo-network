#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from datetime import datetime
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

IMG_LINE = re.compile(r"^(?:\S+/)?\S+\.(?:png|jpe?g)\s*:$", re.I)
CARD_LINE = re.compile(r"[Cc]ard[s]?\s*([1-3]|one|two|three)\s*:")


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
        if not ln or ln.startswith("#"):
            continue
        m = CARD_LINE.match(ln)
        if m:
            raw = m.group(1)
            cur = (int(raw) if raw.isdigit()
                   else {"one": 1, "two": 2, "three": 3}[raw])
            cards[cur] = []
            continue
        if cur is not None:
            cards[cur] += [tok_int(x) for x in ln.split(",")]
    return cards


def parse_dictation(lines):
    blocks = []
    cur_img = None
    cur_lines = []
    for ln in lines:
        if IMG_LINE.match(ln.strip()):
            if cur_img:
                blocks.append((cur_img, cur_lines))
            cur_img = ln.strip().rstrip(":")
            cur_lines = []
        else:
            cur_lines.append(ln)
    if cur_img:
        blocks.append((cur_img, cur_lines))
    return blocks


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
    print(f"wrote {dst.name} ({dst.stat().st_size // 1024}KB)")


def next_sheet_number(gt, phone):
    txt = gt.read_text() if gt.exists() else ""
    nums = [int(m) for m in re.findall(r"#blue_sheet_(\d+):", txt)]
    nums += [int(m.group(1)) for p in phone.glob("*.jpeg")
             if (m := re.match(r"blue_sheet_(\d+)", p.name))]
    return (max(nums) + 1) if nums else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Dictated bingo sheet -> "
                                             "phone_sheets jpeg + truth block")
    ap.add_argument("--src", default="manually_dictated_cards.txt")
    ap.add_argument("--gt", default="scan_card_numbers.txt")
    ap.add_argument("--phone", default="phone_sheets")
    ap.add_argument("--start", type=int, default=None,
                    help="first sheet number (default: next free number)")
    ap.add_argument("--no-delete", action="store_true",
                    help="keep source image files after conversion")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and report only; write nothing")
    args = ap.parse_args(argv)

    base = Path(__file__).resolve().parent.parent
    src = Path(args.src)
    if not src.is_absolute():
        src = base / "01_CNN_refactor" / src
    gt = Path(args.gt)
    if not gt.is_absolute():
        gt = base / gt
    phone = Path(args.phone)
    if not phone.is_absolute():
        phone = base / "01_CNN_refactor" / phone
    phone.mkdir(parents=True, exist_ok=True)

    lines = src.read_text().splitlines()
    blocks = parse_dictation(lines)
    if not blocks:
        sys.exit(f"no image headers found in {src}")

    start = args.start if args.start is not None else next_sheet_number(gt, phone)
    if start < 25:
        print(f"WARNING: sheets {start} and up start in train split "
              "(1-24); use --start 25 to keep them held-out", file=sys.stderr)

    gt_text = gt.read_text() if gt.exists() else ""
    sheet_id = start
    out_blocks = []
    for img, sheet_lines in blocks:
        cards = parse_sheet(sheet_lines)
        if not cards:
            sys.exit(f"{img}: no cards parsed")
        for cid in cards:
            assert len(cards[cid]) == 24, (
                f"{img} card{cid}: got {len(cards[cid])} numbers, want 24")
        while re.search(rf"#blue_sheet_{sheet_id:03d}:", gt_text):
            sheet_id += 1
        jpeg = phone / f"blue_sheet_{sheet_id:03d}.jpeg"
        if jpeg.exists():
            sys.exit(f"{jpeg.name} already exists; use --start to skip past it")
        ipath = Path(img)
        if not ipath.is_absolute():
            ipath = src.parent / img
        if not args.dry_run:
            png_to_jpeg(ipath, jpeg)
        blk = [f"#blue_sheet_{sheet_id:03d}:"]
        for cid in sorted(cards):
            grid = column_major_to_grid(cards[cid])
            blk.append(f"#card{cid}")
            for row in grid:
                blk.append("[" + ", ".join(str(v) for v in row) + "]")
            blk.append("")
        out = "\n".join(blk).rstrip() + "\n"
        out_blocks.append(out)
        print(f"\n== {img} -> {jpeg.name} ==\n{out}")
        sheet_id += 1

    if args.dry_run:
        print(f"\ndry-run: {len(out_blocks)} sheet(s), starting at {start}")
        return

    bak = gt.with_name(f"{gt.stem}.bak_dict{datetime.now():%Y-%m-%d_%H%M%S}")
    shutil.copy2(gt, bak)
    with gt.open("a") as f:
        f.write("\n" + "\n".join(out_blocks))
    print("\nappended to", gt.name, f"(backup {bak.name})")

    if not args.no_delete:
        for img, _ in blocks:
            ipath = Path(img)
            if not ipath.is_absolute():
                ipath = src.parent / img
            ipath.unlink(missing_ok=True)
            print("removed source", ipath)


if __name__ == "__main__":
    main(sys.argv[1:])