"""Validate phone-photo sheet floors against the flatbed-scan ground truth.

The user photographs the SAME 30 printed sheets that were flatbed-scanned
and dictated in scan_card_numbers.txt. Because the sheets are identical,
the labels transfer: photo name blue_sheet_NNN -> GT sheet NNN.

Photos are:
  1. decoded EXIF-correctly (cv2 ignores EXIF rotation),
  2. downscaled to the scan extractor's validated scale (width ~1257),
  3. exposure-normalized (color-preserving YUV stretch) to restore the
     bright-paper assumption the scan detector's fixed thresholds rely on,
  then run through extract_scan_cells.sheet_to_cells.

Two checks per sheet:
1. GEOMETRY: are all 25 cells located (per card)?
2. ALIGNMENT proxy: read each cell with the existing scan-domain reader
   (digit_classifier.pth, 99.9% cell-accurate on the scan sheets). The same
   physical sheets photographed in good lighting should read nearly as
   well IF geometry is right; low accuracy means misaligned cells, and the
   mismatched cells are dumped as ASCII ink maps so the user can eyeball
   whether geometry (not the reader) is at fault.

Debug overlays (grid boxes drawn over the normalized sheet) are written to
/tmp/phone_sheet_debug for the user to eyeball.
"""

import ast
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from build_scan_dataset import clean  # noqa: E402
from evaluate_scan_cells import read_cell as scan_read_cell, parse_truth  # noqa: E402
from extract_scan_cells import sheet_to_cells  # noqa: E402
from photo_sheet_cells import sheet_to_cells_teal  # noqa: E402
from predict_digit import DigitClassifier  # noqa: E402

SCAN_WIDTH = 1257
COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]
OUT = Path("/tmp/phone_sheet_debug")


def load_photo(path):
    """EXIF-correct decode to BGR (cv2 ignores EXIF rotation)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def normalize(img):
    """Downscale to scan scale; stretch luminance so paper is bright while
    preserving hue (teal band detection needs color)."""
    H, W = img.shape[:2]
    scale = SCAN_WIDTH / W
    r = img
    if abs(scale - 1.0) > 0.02:
        r = cv2.resize(img, (SCAN_WIDTH, int(round(H * scale))),
                       interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32)
    lo, hi = np.percentile(y, (2, 98))
    y2 = np.clip((y - lo) * (255.0 / max(1.0, hi - lo)), 0, 255)
    yuv[:, :, 0] = y2.astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def ascii_art(cell, w=22, h=12):
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    return "\n".join(
        "".join("#" if v < 150 else (":" if v < 195 else ".") for v in row)
        for row in small)


def main():
    targets = sys.argv[1:]
    if not targets:
        print("usage: evaluate_photo_sheets.py <sheet_dir|photo...>")
        return
    paths = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            for ext in ("jpeg", "jpg", "png"):
                paths += sorted(p.glob(f"*.{ext}"))
        else:
            paths.append(p)

    truth = parse_truth()
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    model.load_state_dict(
        torch.load(Path(__file__).parent / "digit_classifier.pth",
                   map_location=device))
    model.eval()

    g_cells = g_ok = 0
    for path in paths:
        m = re.search(r"(\d+)(?=\.(?:jpe?g|png)$)", path.name)
        if not m:
            print(f"{path.name}: no sheet number in name, skip")
            continue
        s = int(m.group(1))
        try:
            img = load_photo(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name}: unreadable ({exc})")
            continue
        img = normalize(img)
        cells = sheet_to_cells_teal(img, return_geometry=True)
        if not cells:
            cells = sheet_to_cells(img, return_geometry=True)
        if not cells:
            print(f"{path.name}: no teal bands / cells detected")
            continue
        overlay = img.copy()
        by_card = {}
        for cid, r, c, cell, box in cells:
            by_card.setdefault(cid, {})[(r, c)] = (cell, box)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]),
                          (0, 200, 0), 1)
        cv2.imwrite(str(OUT / f"overlay_{Path(path).stem}.jpg"), overlay)
        print(f"== {path.name}: sheet {s}, cards {sorted(by_card)} "
              f"(overlay -> {OUT / ('overlay_' + Path(path).stem + '.jpg')})")
        for cid in sorted(by_card):
            gt = truth.get((s, cid))
            if gt is None:
                print(f"  card{cid}: no GT for sheet {s}")
                continue
            ok = 0
            tot_c = 0
            for r in range(5):
                for c in range(5):
                    if r == 2 and c == 2:
                        continue
                    ent = by_card[cid].get((r, c))
                    if ent is None:
                        print(f"    card{cid} r{r}c{c}: NO CELL")
                        continue
                    cell, _ = ent
                    tot_c += 1
                    g_cells += 1
                    target = gt[r][c]
                    num, conf, _ = scan_read_cell(
                        model, clean(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)),
                        COLUMN_RANGES[c])
                    if num == target:
                        ok += 1
                        g_ok += 1
                    else:
                        print(f"    card{cid} r{r}c{c} GT={target} "
                              f"read={num} ({conf:.2f})")
                        print(ascii_art(cell))
            print(f"  card{cid}: {ok}/{tot_c}")

    print(f"\nALIGNMENT PROXY (scan reader on phone cells): "
          f"{g_ok}/{g_cells} = {g_ok / g_cells:.4f}")


if __name__ == "__main__":
    main()