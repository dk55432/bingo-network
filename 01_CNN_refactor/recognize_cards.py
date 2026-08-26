"""
End-to-end card reader: photo -> cells -> digit glyphs -> numbers.

Attaches the trained DigitClassifier (predict_digit.py /
digit_classifier.pth) to the cell extraction of parse_bingo_sheet.py.
The cell -> glyph splitter ports find_digit_components from
00_card_scan_refactor/extract_digits.py, the same logic that built
digits/train, so the classifier sees input shaped like its training
data.

Bingo column ranges (B1-15, I16-30, N31-45, G46-60, O61-75) give a free
sanity check on every recognized card.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from predict_digit import DigitClassifier, test_transform
from parse_bingo_sheet import (
    deskew,
    estimate_skew_angle,
    _deskew_header,
    extract_grid_cells,
)

COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]

MODEL_PATH = Path(__file__).parent / "digit_classifier.pth"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def _pen_mask(cell):
    """
    Boolean mask of marker ink (green daub rings, red pens).

    The training set came from clean 1-3 card close-ups, so its glyphs
    were never contaminated by daub; on 6-card sheets most played cells
    carry a green ring whose GRAYSCALE value sits below the digit
    threshold (~90-130), fusing it into the digit component. Hue/saturation
    separates marker ink from black print regardless of darkness.
    """
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h_ >= 35) & (h_ <= 95)
    red = (h_ <= 10) | (h_ >= 160)
    return ((green | red) & (s_ > 60)).astype(np.uint8)


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DigitClassifier().to(_DEVICE)
        _MODEL.load_state_dict(
            torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=True)
        )
        _MODEL.eval()
    return _MODEL


def _cut_wide(comp_box, mask):
    """Split a two-digits-touching component at its density valley."""
    X, Y, W, H = comp_box
    sub = mask[Y:Y + H, X:X + W]
    col_ink = sub.sum(axis=0).astype(np.float32) / 255.0
    lo, hi = int(0.30 * W), int(0.70 * W)
    if hi - lo < 4:
        return [comp_box]
    cut = X + lo + int(np.argmin(col_ink[lo:hi]))
    if cut - X < 6 or X + W - cut < 6:
        return [comp_box]
    return [(X, Y, cut - X, H), (cut, Y, X + W - cut, H)]


def split_digits(cell):
    """
    Split one cell crop into left-to-right digit glyph images.

    Ports extract_digits.find_digit_components (dark-pixel mask, noise
    removal, connected components — the logic that built digits/train)
    with two additions the levelled cells need: printed rules are
    subtracted before labelling so digits fused to them survive, and a
    component wider than one digit is cut at its ink valley ('19' with
    touching glyphs labels as ONE component otherwise).
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Marker exclusion FIRST (see _pen_mask): daub rings are dark in
    # grayscale and otherwise fuse with digit strokes.
    pen = cv2.dilate(_pen_mask(cell),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # Local-contrast print mask (same idea as parse_bingo_sheet's
    # pmc): a global threshold floods with green/red daub ink, whose
    # smears then shred digits during rule removal. The local median
    # tracks the daub background, so only true dark strokes remain.
    k = max(5, (min(h, w) // 4) * 2 + 1)
    lm = cv2.medianBlur(gray, k).astype(np.int16)
    mask = ((gray.astype(np.int16) < lm - 35) & (pen == 0)) \
        .astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (3, 3)))

    # Subtract printed rules (horizontal + vertical) so digits sitting
    # on them do not fuse into giant components. Kernel lengths must be
    # LONGER than a full two-digit number (~0.6*w / ~0.7*h) — anything
    # shorter erases the numbers themselves ('19' with touching glyphs,
    # '1' stems) instead of the rules.
    hker = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(30, int(0.75 * w)), 1))
    vker = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(26, int(0.72 * h))))
    rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hker)
    rules |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, vker)
    rules = cv2.dilate(rules, cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3)))
    clean = cv2.bitwise_and(mask, cv2.bitwise_not(rules))

    # Inner region only: outermost pixels belong to grid rules and
    # neighbour-cell bleed, not this cell's digits.
    mx, my = max(3, int(0.10 * w)), max(3, int(0.10 * h))
    work = clean[my:h - my, mx:w - mx]

    num, _, stats, _ = cv2.connectedComponentsWithStats(work, 8)

    comps = []
    for i in range(1, num):
        X, Y, W, H, A = stats[i]
        # Minimums do the junk rejection: vertical rule slivers are
        # 2-3px wide, horizontal stubs fail the height floor. Real
        # digits often TOUCH the cell's baseline rule, so components
        # must NOT be rejected for reaching the window edge.
        if A < 25 or H < max(8, int(0.25 * (h - 2 * my))) or W < 4:
            continue
        if W > 0.62 * w or A < 0.10 * W * H:
            continue
        # Shift back to full-cell coordinates.
        comps.append((X + mx, Y + my, W, H))

    if not comps:
        return []

    # Wide components are touching digit pairs — split them. Split
    # pieces carry split=True so the merge step cannot fuse them back
    # together (their gap is zero by construction).
    widened = []
    for comp in comps:
        if comp[2] > 0.42 * w:
            widened += [(b[0], b[1], b[2], b[3]) for b in _cut_wide(comp,
                                                                    clean)]
        else:
            widened.append(comp)
    split_flags = len(widened) != len(comps)
    comps = widened

    # Merge remaining close pairs (a skinny '1' beside its tens digit),
    # never touching valley-split pieces.
    comps.sort(key=lambda c: c[0])
    merged = []
    for comp in comps:
        placed = False
        if merged and not split_flags:
            px, py, pw, ph = merged[-1]
            X, Y, W, H = comp
            if X - (px + pw) <= max(5, int(0.30 * min(W, pw))) \
                    and Y < py + ph and Y + H > py:
                nx0, ny0 = min(px, X), min(py, Y)
                nx1, ny1 = max(px + pw, X + W), max(py + ph, Y + H)
                merged[-1] = [nx0, ny0, nx1 - nx0, ny1 - ny0]
                placed = True
        if not placed:
            merged.append(list(comp))

    merged = sorted(merged, key=lambda m: m[0])[:2]

    glyphs = []
    for X, Y, W, H in merged:
        glyphs.append(cell[Y:Y + H, X:X + W])
    return glyphs


def _render_glyph(glyph):
    """
    Normalise one glyph crop: global <150 ink mask (matching the
    training data's digit_preprocess recipe) with pen exclusion via
    HSV colour, largest connected component, tight grayscale crop,
    aspect-preserving fit into 22x22, centred on a 28x28 white canvas.

    split_digits uses local-contrast to robustly FIND glyph boxes
    (handles daub), but by the time we have a tight crop the digit is
    the dominant dark feature. Global <150 matches training; local-
    contrast with large k kills everything on small crops.
    """
    if glyph.ndim == 2:
        gray = glyph
        pen = np.zeros_like(gray)
    else:
        gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
        pen = _pen_mask(glyph)
    h, w = gray.shape
    pen = cv2.dilate(pen,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = ((gray < 150) & (pen == 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (3, 3)))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return None
    # Prefer a component that does NOT touch the crop boundary — those
    # are gridline slivers or card-edge artefacts, not digits.
    best = None
    best_area = 0
    for i in range(1, num):
        X, Y, W, H, A = stats[i]
        touches = (X == 0 or Y == 0
                   or X + W >= w - 1 or Y + H >= h - 1)
        if not touches and A > best_area:
            best_area = A
            best = i
    if best is None:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == best)
    if len(ys) < 4 or ys.max() == ys.min() or xs.max() == xs.min():
        return None
    # White background + original grayscale strokes (training data has
    # clean white backgrounds; raw cell JPEGs have gray haze).
    clean = np.full_like(gray, 255)
    clean[ys, xs] = gray[ys, xs]
    digit = clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Aspect-preserving thumbnail into 22x22 (PIL.thumbnail semantics).
    dh, dw = digit.shape
    scale = min(22.0 / dw, 22.0 / dh, 1.0)
    nw, nh = max(1, int(round(dw * scale))), max(1, int(round(dh * scale)))
    small = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((28, 28), 255, np.uint8)
    x0 = (28 - nw) // 2
    y0 = (28 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def classify_digit(model, glyph):
    """Return (digit, confidence) for one grayscale glyph crop."""
    canvas = _render_glyph(glyph)
    if canvas is None:
        return -1, 0.0
    pil = Image.fromarray(canvas)
    tensor = test_transform(pil).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    conf, pred = probs.max(dim=0)
    return int(pred.item()), float(conf.item())


def _predict_from_raw(model, cell_gray):
    """Predict a bingo number from a raw cell image using the digit CNN.

    Tries both full-cell and left/right split, returns whichever has
    higher confidence. Matches the training data convention where
    two-digit numbers were split left/right.
    """
    h, w = cell_gray.shape
    pil = Image.fromarray(cell_gray)

    t_full = test_transform(pil).unsqueeze(0).to(_DEVICE)
    t_left = test_transform(pil.crop((0, 0, w // 2, h))).unsqueeze(0).to(_DEVICE)
    t_right = test_transform(pil.crop((w // 2, 0, w, h))).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        p_full = torch.softmax(model(t_full), dim=1)[0]
        p_left = torch.softmax(model(t_left), dim=1)[0]
        p_right = torch.softmax(model(t_right), dim=1)[0]

    full_pred = p_full.argmax().item()
    full_conf = p_full.max().item()

    left_pred = p_left.argmax().item()
    right_pred = p_right.argmax().item()
    two_conf = (p_left.max().item() + p_right.max().item()) / 2

    if full_conf >= two_conf or left_pred == 0:
        return full_pred, full_conf, 1
    return 10 * left_pred + right_pred, two_conf, 2


def read_cell(model, cell):
    """Read one cell using raw cell resize. Returns (number, confidence, n_glyphs)."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    return _predict_from_raw(model, gray)


def read_card(card):
    """
    Recognize one levelled card crop.

    Returns dict with the 5x5 number grid (None where unread), the raw
    glyph counts, confidences, and column-range violation flags.
    """
    angle = estimate_skew_angle(card)
    card = deskew(card)
    rot = _deskew_header(card)
    if rot is not card:
        card = rot

    cells, x_lines, y_lines, row_cuts, fitted, col_rows = \
        extract_grid_cells(card)
    result = {
        "fitted": fitted,
        "numbers": [[None] * 5 for _ in range(5)],
        "confidences": [[0.0] * 5 for _ in range(5)],
        "violations": [],
    }
    if not fitted:
        return result

    model = load_model()
    for r in range(5):
        for c in range(5):
            num, conf, nglyphs = read_cell(model, cells[r][c])
            result["numbers"][r][c] = num
            result["confidences"][r][c] = conf
            if nglyphs > 2:
                result["violations"].append(
                    f"r{r + 1}c{c + 1}: {nglyphs} glyphs")
            elif num is None and nglyphs == 0:
                result["violations"].append(f"r{r + 1}c{c + 1}: no digits")
            elif num is not None and \
                    not (COLUMN_RANGES[c][0] <= num <= COLUMN_RANGES[c][1]):
                result["violations"].append(
                    f"r{r + 1}c{c + 1}: {num} outside "
                    f"{COLUMN_RANGES[c]}")
    return result


def format_card(result):
    if not result["fitted"]:
        return "  (grid fit failed)"
    rows = []
    for r in range(5):
        rows.append("  " + " ".join(
            f"{result['numbers'][r][c]:>2}"
            if result["numbers"][r][c] is not None else " ."
            for c in range(5)))
    mean_conf = float(np.mean([result["confidences"][r][c]
                               for r in range(5) for c in range(5)
                               if result["numbers"][r][c] is not None]))
    out = "\n".join(rows)
    if result["violations"]:
        out += "\n  flags: " + "; ".join(result["violations"][:8])
    out += f"\n  mean conf: {mean_conf:.2f}"
    return out


def main():
    import sys
    targets = sys.argv[1:] or ["sample_bingo_cards"]
    paths = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            paths += sorted(p.glob("*.jpeg")) + sorted(p.glob("*.jpg"))
        else:
            paths.append(p)

    total_cards = flagged = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        try:
            cards, _, _ = __import__(
                "parse_bingo_sheet").extract_card_grids(image)
        except Exception as exc:  # noqa: BLE001
            print(f"{path.stem}: detection failed ({exc})")
            continue
        print(f"{path.stem}: {len(cards)} card(s)")
        for i, card in enumerate(cards):
            res = read_card(card)
            total_cards += 1
            mark = ""
            if res["fitted"] and res["violations"]:
                flagged += 1
                mark = f"  [{len(res['violations'])} flags]"
            print(f"  card_{i + 1}{mark}")
            if res["fitted"]:
                print(format_card(res))
    print(f"\n{total_cards} cards, {flagged} with flags")


if __name__ == "__main__":
    main()
