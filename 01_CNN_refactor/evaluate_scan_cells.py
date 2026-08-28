"""End-to-end scan cell -> number evaluation with unknown digit count.

For each cell (from /tmp/scan_ds/{train,valid}/cells), extract the ink
bounding region, try both single-digit and two-digit interpretations
(mid-third valley split), and pick the higher-confidence reading.
Compares against scan_card_numbers.txt ground truth at number level.
"""

import ast
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from predict_digit import DigitClassifier

GT = "/Users/davidkohn/Downloads/bingo-network/scan_card_numbers.txt"
DS = "/tmp/scan_ds"

test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]


def parse_truth():
    truth = {}
    sheet = card = None
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
    """Ink bbox of the largest component, ignoring small rule slivers."""
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


def classify_glyph(model, gray):
    if gray is None or gray.size == 0 or gray.shape[0] == 0 or gray.shape[1] == 0:
        return -1, 0.0
    pil = Image.fromarray(gray)
    t = test_tf(pil).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(t), dim=1)[0]
    return int(p.argmax().item()), float(p.max().item())


def valley_split(model, gray, x0, y0, x1, y1, bw):
    """If a single component looks like two fused digits, split at the
    deepest column-profile valley in the middle band. Returns (num, conf, 2)
    or None if no convincing valley.

    Digits are wide (bb >= 0.45*cell width for a blackletter-style '1'/'4'),
    so the trigger is a genuinely empty gap: a column whose ink is <6% of the
    maximum column ink, flanked on both sides (within 5 px windows) by at
    least 30% of the maximum. Two fused digits leave such a gap; a single
    digit never does.
    """
    sub = gray[y0:y1 + 1, x0:x1 + 1]
    bin = (sub < 170).astype(np.uint8)
    prof = bin.sum(axis=0).astype(float)
    maxprof = float(prof.max())
    if maxprof <= 0:
        return None
    best = None
    best_score = maxprof
    mid_lo = int(0.25 * bw)
    mid_hi = int(0.75 * bw)
    for v in range(mid_lo + 1, mid_hi):
        if prof[v] > 0.06 * maxprof:
            continue
        L = float(prof[v - 5:v].sum()) if v >= 5 else 0.0
        R = float(prof[v + 1:v + 6].sum())
        if L < 0.30 * maxprof and R < 0.30 * maxprof:
            continue
        if prof[v] < best_score:
            best_score = prof[v]
            best = v
    if best is None:
        return None
    v = best
    left = tight(sub[:, :v])
    right = tight(sub[:, v:])
    if left is None or right is None or left.size == 0 or right.size == 0 or \
       left.shape[0] < 8 or right.shape[0] < 8:
        return None
    dl, cl = classify_glyph(model, left)
    dr, cr = classify_glyph(model, right)
    if dl < 0 or dr < 0:
        return None
    return 10 * dl + dr, (cl + cr) / 2, 2


def read_cell(model, gray, col_range=None):
    """Split by components, read each glyph, assemble the number.

    col_range (lo, hi) is the bingo column's authorized number range; a
    fused two-digit split is only accepted when its value lands in range
    (single digits never do for columns N/G/O, and fused pairs with a
    leading digit do for B/I), which cleanly separates fused "44" from a
    wide-but-single "7".
    """
    bin = (gray < 170).astype(np.uint8)
    ys, xs = np.nonzero(bin)
    if len(ys) == 0:
        return None, 0.0, 1
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        bin[y0:y1 + 1, x0:x1 + 1], 8)
    comps = [stats[i] for i in range(1, n)
             if stats[i][cv2.CC_STAT_AREA] > 0.02 * (bh * bw)
             and stats[i][cv2.CC_STAT_HEIGHT] >= 0.30 * bh]

    whole = tight(gray[y0:y1 + 1, x0:x1 + 1])

    if len(comps) == 1:
        # single wide glyph may be two digits fused (e.g. "44"); try a
        # column-profile valley split. Accept it only when (a) the resulting
        # number is legal for the bingo column, and (b) either the single
        # reading is itself illegal in that column, or the split is strictly
        # more confident -- so a legal single digit is never sacrificed for
        # an equally-legal (but wrong) split.
        if bw > 0.45 * gray.shape[1]:
            d1, c1 = classify_glyph(model, whole)
            result = valley_split(model, gray, x0, y0, x1, y1, bw)
            if result is not None and col_range is not None:
                val, conf, _ = result
                if col_range[0] <= val <= col_range[1] and \
                   (d1 not in range(col_range[0], col_range[1] + 1)
                        or conf > c1):
                    return result
            return d1, c1, 1
        d, c = classify_glyph(model, whole)
        return d, c, 1

    # two glyphs: split at the midpoint of the boundary between the two
    # dominant components (works even when a rule bridge overlaps them).
    comps.sort(key=lambda c: c[0])
    if len(comps) < 2:
        d, c = classify_glyph(model, whole)
        return d, c, 1
    a, b = comps[:2]
    a_r = a[0] + a[2]
    b_l = b[0]
    v = x0 + (a_r + b_l) // 2
    if v - x0 < 4 or x1 - v < 4:
        d, c = classify_glyph(model, whole)
        return d, c, 1
    left = tight(gray[y0:y1 + 1, x0:v])
    right = tight(gray[y0:y1 + 1, v:x1 + 1])
    dl, cl = classify_glyph(model, left)
    dr, cr = classify_glyph(model, right)
    return 10 * dl + dr, (cl + cr) / 2, 2


def main():
    truth = parse_truth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    model.load_state_dict(
        torch.load("digit_classifier.pth", map_location=device,
                   weights_only=True))
    model.eval()

    split = "valid"
    cells = sorted(os.listdir(os.path.join(DS, split, "cells")))
    correct = total = 0
    away = 0
    wrong_details = []
    per_sheet = {}
    truncated = {"correct": 0, "total": 0}
    for fn in cells:
        m = re.match(r"blue_sheet_0*(\d+)_card(\d)_r(\d)c(\d)", fn)
        sheet, card, r, c = map(int, m.groups())
        grid = truth.get((sheet, card))
        if grid is None:
            continue
        num = grid[r][c]
        if num == "FREE":
            continue
        gray = cv2.imread(os.path.join(DS, split, "cells", fn),
                          cv2.IMREAD_GRAYSCALE)
        pred, conf, ng = read_cell(model, gray, COLUMN_RANGES[c])
        num = int(num)
        is_trunc = (card == 3 and r == 4)
        if is_trunc:
            truncated["total"] += 1
            truncated["correct"] += 1 if pred == num else 0
            continue
        total += 1
        if pred == num:
            correct += 1
            per_sheet.setdefault(sheet, [0, 0])[0] += 1
        else:
            away += abs(pred - num) if pred is not None else 999
            per_sheet.setdefault(sheet, [0, 0])[1] += 1
            wrong_details.append((sheet, card, r, c, num, pred, conf, ng))

    print(f"valid (excl card3 r4): {correct}/{total} = {correct/total:.3f}")
    print(f"card3 r4 truncated: "
          f"{truncated['correct']}/{truncated['total']} = "
          f"{truncated['correct']/max(1,truncated['total']):.3f}")
    print(f"mean abs error (wrong only): {away/max(1, total-correct):.1f}")
    for sheet in sorted(per_sheet):
        ok, bad = per_sheet[sheet]
        print(f"  sheet {sheet:03d}: {ok}/{ok+bad}")
    print("\nwrong cells (sheet, card, r, c, truth, pred, conf, nglyphs):")
    for w in wrong_details[:40]:
        print(" ", w)


if __name__ == "__main__":
    main()