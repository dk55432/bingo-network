"""Build the phone-domain glyph dataset for training/eval.

The labeled phone glyphs live in 00_card_scan_refactor/digits/{train,test}
at 00_card_scan_refactor/digits. They come in two forms: already-normalized
28x28 white-canvas glyphs and larger native bbox crops. Every sample is run
through recognize_cards._render_glyph (tight crop -> aspect-preserving fit
into 22x22 centered on a 28x28 white canvas) so the training, validation,
and inference-time inputs are exactly the same representation.

Output: /tmp/phone_digits_dir/{train,valid}/{0..9}/*.jpg
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from recognize_cards import _render_glyph  # noqa: E402

SRC = Path("/Users/davidkohn/Downloads/bingo-network/00_card_scan_refactor/digits")
DST = Path("/tmp/phone_digits_dir")


def process(src_root, dst_root):
    total = skipped = 0
    for cls in map(str, range(10)):
        src_dir = src_root / cls
        dst_dir = dst_root / cls
        if not src_dir.is_dir():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.glob("*")):
            if not f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            canvas = _render_glyph(img)
            if canvas is None:
                skipped += 1
                continue
            cv2.imwrite(str(dst_dir / f.name), canvas)
            total += 1
    return total, skipped


if __name__ == "__main__":
    for split in ("train", "valid"):
        dst = DST / split
        dst.mkdir(parents=True, exist_ok=True)
        src = SRC / ("train" if split == "train" else "test")
        total, skipped = process(src, dst)
        print(f"{split}: {total} rendered, {skipped} skipped/unreadable")
        for cls in map(str, range(10)):
            n = len(list((dst / cls).glob("*"))) if (dst / cls).is_dir() else 0
            print(f"  class {cls}: {n}")