"""Extract 25 bingo cells from flatbed scan sheets (3 cards per sheet).

Scan layout (validated across sheets 001-003):
- portrait sheet, 3 cards stacked vertically
- each card: teal BINGO header band (~35% teal rows), then 5x5 number grid
- grid: 5 number rows, interior horizontal grid lines at ~192px pitch
- grid lines are thin dark lines spanning the card width
- the 3rd card may be cut at the bottom of the scan

Pipeline per sheet:
1. detect full-width dark rows -> candidate horizontal grid lines across sheet
2. detect teal header bands -> card tops (row 1 top boundary)
3. for each card, group the horizontal lines into the 5 rows
4. detect vertical grid columns within the card's row span
5. extract 5x5 cells, save as JPEG
"""

import argparse, os
import cv2
import numpy as np

H_LINE_FRAC = 0.55      # horizontal grid line: dark run >= 55% of width
V_LINE_FRAC = 0.45      # vertical grid line: dark run >= 45% of card height
ROW_PITCH = 192
INSET = 5               # px to trim from each cell edge (exclude grid lines)


def max_run(v):
    nz = v > 0
    best = cur = 0
    for b in nz:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def cluster(vals, sep=10):
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= sep:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(np.mean(c)) for c in out]


def find_teal_bands(img):
    """Return list of (start, end) y ranges of teal header bands.

    These sheets have a teal (hue ~100) BINGO header band per card; the
    number rows below have low saturation, so a high per-row saturation
    fraction isolates the header bands. The band END is the card's row-1 top.
    """
    return _mask_bands(_band_mask(img, hrange=(90, 125), s_min=60, v_min=100))


def find_header_bands(img):
    """find_teal_bands, generalized to any header-band color.

    Bingo sheet sets come in different print colors (teal/blue sheets,
    orange sets, green, ...): each card has a saturated horizontal BINGO
    banner and the number rows below are low-saturation.  Pick the color
    family producing the most band-like rows (teal first so existing
    blue-sheet behavior is unchanged) and return its bands.
    """
    spec = _header_hue_spec(img)
    if spec is None:
        return []
    hl, hh, s_min, v_min, rowfrac = spec
    return _mask_bands(_band_mask(img, hrange=(hl, hh), s_min=s_min,
                                  v_min=v_min), rowfrac)


# Candidate header print colors, teal first (original blue sheets).  Each
# entry is (hue range, s_min, v_min, row_fraction).  Teal keeps the
# strict original thresholds so blue-sheet detection is byte-for-byte
# unchanged; the other families relax them because e.g. green header ink
# prints dimmer/less saturated than teal.
_BAND_FAMILIES = [
    ((90, 125), 60, 100, 0.20),  # teal (original blue sheets)
    ((125, 160), 60, 100, 0.20),  # mid blues
    ((5, 30), 80, 100, 0.15),     # orange sets
    ((30, 90), 45, 45, 0.18),     # green/yellow sets (dimmer ink)
    ((0, 12), 40, 60, 0.10),      # pink/red sets (pale, low-ish sat)
    ((160, 180), 60, 80, 0.15),   # magenta / purples
]


def _header_hue_spec(img):
    """Return the (h_low, h_high, s_min, v_min, rowfrac) of the sheet's
    header color, or None if no sheet-like colored bands exist.

    Teal is the default and always wins when it yields any bands — the
    original blue sheets must remain byte-for-byte identical.  For the
    other print colors (orange/green/pink sets, ...), prefer whichever
    family produces a *structurally plausible* sheet: exactly three
    evenly-spaced, similarly-sized bands (a full strip — the normal case),
    else a single band (single-card photo).  Structural plausibility stops
    unrelated colored objects in the frame (notebooks, mugs) from being
    mistaken for card headers.  If nothing is plausible, fall back to the
    highest raw band score, then to the image's own dominant saturated
    hue so unseen print colors still work."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int)
    s = hsv[:, :, 1].astype(int)
    v = hsv[:, :, 2].astype(int)
    teal_mask = _hue_in(h, 90, 125) & (s > 60) & (v > 100)
    teal_bands = _mask_bands(teal_mask)
    teal_score, teal_ok = _sheet_structure(teal_bands, img.shape[0])
    # Strict teal wins only if it yields a valid 3-card structure.
    # If only 2 bands found (or structure invalid), try relaxed thresholds
    # to catch faded 3rd headers.
    if teal_ok and (len(teal_bands) == 3 or teal_score >= 100):
        return (90, 125, 60, 100, 0.20)
    # Try relaxed teal thresholds for faded 3rd headers (or 2-card sheets)
    for s_min, v_min in [(45, 80), (30, 50), (15, 30)]:
        relaxed_mask = _hue_in(h, 90, 125) & (s > s_min) & (v > v_min)
        relaxed_bands = _mask_bands(relaxed_mask, 0.20)
        score, ok = _sheet_structure(relaxed_bands, img.shape[0])
        # For relaxed thresholds, accept 2 or 3 bands as valid structures
        # (2 bands = 2-card sheet, 3 bands = 3-card sheet)
        # _sheet_structure returns ok=False for 2 bands, so we check manually
        n_bands = len(relaxed_bands)
        if (ok and n_bands >= 1) or (not ok and n_bands in (2, 3)):
            return (90, 125, s_min, v_min, 0.20)
    # Fall through to other color families
    candidates = []
    for idx, ((hl, hh), s_min, v_min, rowfrac) in enumerate(_BAND_FAMILIES[1:]):
        bands = _mask_bands(_hue_in(h, hl, hh) & (s > s_min) & (v > v_min),
                            rowfrac)
        score, ok = _sheet_structure(bands, img.shape[0])
        if ok:
            # (structure score, family priority, spec) — earlier families win ties
            candidates.append((score, -idx, (hl, hh, s_min, v_min, rowfrac)))
    if candidates:
        return max(candidates)[2]
    # Try relaxed thresholds for other color families (low-light robustness)
    # Similar to teal's relaxed attempts, but for all non-teal families.
    relaxed_candidates = []
    for idx, ((hl, hh), s_min, v_min, rowfrac) in enumerate(_BAND_FAMILIES[1:]):
        best_for_family = None
        best_n_bands = 0
        best_spec = None
        for s_min, v_min in [(60, 80), (40, 60), (30, 40), (20, 30), (15, 20)]:
            relaxed_mask = _hue_in(h, hl, hh) & (s > s_min) & (v > v_min)
            relaxed_bands = _mask_bands(relaxed_mask, rowfrac)
            filtered_bands = [b for b in relaxed_bands if b[0] > 0]
            score, ok = _sheet_structure(filtered_bands, img.shape[0])
            n_bands = len(filtered_bands)
            # Accept if structurally valid OR if we have 2-3 bands (like teal's relaxed logic)
            # This handles gray sheets where headers are faint and bands may not be perfectly even
            if ok or (not ok and n_bands >= 2):
                if n_bands > best_n_bands:
                    best_n_bands = n_bands
                    best_for_family = (hl, hh, s_min, v_min, rowfrac)
        if best_for_family:
            relaxed_candidates.append(best_for_family)
    if relaxed_candidates:
        # Prefer families that detected more cards (more bands)
        relaxed_candidates.sort(key=lambda x: x[0], reverse=True)
        return relaxed_candidates[0]  # Return the (hl, hh, s_min, v_min, rowfrac) tuple

    # Nothing clearly sheet-like yet.  The raw-score and dominant-hue
    # fallbacks exist so unseen print colors still get a chance, but they
    # only count if the bands they produce are themselves structurally
    # plausible (1 or 3 uniform, evenly-spaced bands).  Otherwise return
    # None so the caller raises an actionable retake error — gray sheets
    # photographed small/underlit are genuinely indistinguishable from the
    # warm surroundings, and reading the desk as "cards" is worse than
    # asking for a better photo.
    fallbacks = []
    for (hl, hh), s_min, v_min, rowfrac in _BAND_FAMILIES[1:]:
        bands = _mask_bands(_hue_in(h, hl, hh) & (s > s_min) & (v > v_min),
                            rowfrac)
        score, ok = _sheet_structure(bands, img.shape[0])
        if ok:
            fallbacks.append((score, (hl, hh, s_min, v_min, rowfrac)))
    if fallbacks:
        return max(fallbacks)[1]
    sat = (s > 45) & (v > 45)
    if sat.sum() < 0.01 * img.shape[0] * img.shape[1]:
        return None
    dom = int(np.bincount(h[sat].ravel()).argmax())
    adaptive = ((dom - 20) % 180, (dom + 20) % 180, 45, 45, 0.12)
    bands = _mask_bands(_hue_in(h, dom - 20, dom + 20) & (s > 45) & (v > 45),
                        0.12)
    if _sheet_structure(bands, img.shape[0])[1]:
        return adaptive
    return None


def _sheet_structure(bands, img_height=None):
    """Score a band layout's plausibility as a bingo sheet: 3 evenly
    spaced, similarly sized bands (a full strip) is the strong case;
    1 band (single card) is acceptable.  Returns (score, ok)."""
    if not bands:
        return 0.0, False
    if len(bands) == 1:
        # Reject bands that span most of the image - those are adaptive
        # hue fallbacks capturing the whole frame, not a real header.
        if img_height and (bands[0][1] - bands[0][0]) > 0.5 * img_height:
            return 0.0, False
        return 10.0, True
    if len(bands) == 3:
        tops = sorted(b[1] for b in bands)
        pitches = [tops[i + 1] - tops[i] for i in range(2)]
        heights = [b[1] - b[0] for b in bands]
        if (pitches[0] > 0
                and max(pitches) - min(pitches) <= 0.28 * max(pitches)
                and min(heights) >= 0.45 * max(heights)):
            return 1000.0 + min(heights) / max(heights), True
    return float(len(bands)), False


def _hue_in(h, hl, hh):
    if hl <= hh:
        return (h >= hl) & (h <= hh)
    return (h >= hl) | (h <= hh)


def _band_mask(img, hrange, s_min, v_min):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int)
    s = hsv[:, :, 1].astype(int)
    v = hsv[:, :, 2].astype(int)
    return _hue_in(h, *hrange) & (s > s_min) & (v > v_min)


def _mask_bands(mask, rowfrac=0.20):
    """Rows where the band covers >rowfrac of the width, grouped into
    contiguous runs at least 15 rows tall, as (start, end) y ranges."""
    rowcnt = mask.mean(axis=1)
    rows = [y for y in range(len(rowcnt)) if rowcnt[y] > rowfrac]
    runs = []
    for y in rows:
        if runs and y - runs[-1][-1] <= 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    bands = []
    for r in runs:
        if len(r) >= 15:  # ignore isolated colored specks
            bands.append((r[0], r[-1]))
    return bands


def _band_score(bands):
    return sum(hi - lo for lo, hi in bands) + 1000 * len(bands)


def detect_hlines(bin_img):
    H, W = bin_img.shape
    found = [y for y in range(H) if max_run(bin_img[y, :]) >= H_LINE_FRAC * W]
    return cluster(found, sep=6)


def detect_vlines(bin_img, y0, y1):
    ch = y1 - y0
    if ch <= 0:
        return []
    found = [x for x in range(bin_img.shape[1])
             if max_run(bin_img[y0:y1, x]) >= V_LINE_FRAC * ch]
    return cluster(found, sep=6)


def pick_columns(vlines, width=None):
    """Choose 6 column boundaries (borders + 4 interior) from candidate vlines.

    The interior grid columns are strong and ~225px apart; outer borders are
    faint. Strategy: find the 4 consecutive candidates with the most regular
    ~225px spacing that sit in the card's mid-width (interior columns live
    well inside the page), then extrapolate borders at one pitch either side.
    """
    v = sorted(vlines)
    if not v:
        return None
    pitch = 225.0
    cand = np.diff(v)
    if len(cand) >= 3:
        med = float(np.median(cand))
        if 190 <= med <= 260:
            pitch = med

    lo = 0.10 * (width or 1257)
    hi = 0.92 * (width or 1257)

    best = None
    best_dev = None
    for start in range(0, max(1, len(v) - 3)):
        sub = v[start:start + 4]
        if sub[0] < lo or sub[-1] > hi:
            continue
        g = np.diff(sub)
        dev = float(np.std(g - pitch))
        if best_dev is None or dev < best_dev:
            best_dev = dev
            best = sub
    if best is None:
        return None
    interior = best
    left = interior[0] - pitch
    right = interior[-1] + pitch
    # if a detected candidate is very close to the extrapolated border, use it
    for cand_x in v:
        if 0 <= cand_x <= left + 0.30 * pitch:
            left = cand_x
        if right - 0.30 * pitch <= cand_x:
            right = cand_x
    return [int(left)] + list(interior) + [int(right)]


def sheet_to_cells(img, verbose=False, return_geometry=False):
    """Return list of (card_id, row, col, cell_rgb_image), optionally
    (card_id, row, col, cell_rgb_image, (x0, y0, x1, y1)) when
    return_geometry=True so callers can draw debug overlays."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = (g < 190).astype(np.uint8)
    H, W = img.shape[:2]

    hlines = detect_hlines(bin_img)
    teal_bands = find_teal_bands(img)
    if verbose:
        print(f"    teal bands: {teal_bands}")
        print(f"    hlines: {hlines}")

    cards = []
    for (ts, te) in teal_bands:
        # row-1 top = teal band end; rows 1..5 separated by the h-lines after
        own = [y for y in hlines if y > te + 5][:5]
        if not own:
            continue
        row_b = [te] + own
        # fill remainder (e.g. last card cut off at scan bottom)
        while len(row_b) < 6:
            row_b.append(row_b[-1] + ROW_PITCH)
        cards.append(row_b[:6])
        if verbose:
            print(f"    card rows: {row_b[:6]}")

    cells_out = []
    for cid, row_b in enumerate(cards, 1):
        mid_y0 = row_b[0]
        mid_y1 = row_b[5]
        vlines = detect_vlines(bin_img, mid_y0, mid_y1)
        vr = pick_columns(vlines)
        if vr is None:
            if verbose:
                print(f"    card{cid}: no cols from {vlines}, SKIP")
            continue
        if verbose:
            print(f"    card{cid} cols: {vr}")

        for r in range(5):
            for c in range(5):
                yA, yB = row_b[r] + INSET, row_b[r + 1] - INSET
                xA, xB = vr[c] + INSET, vr[c + 1] - INSET
                cell = img[yA:yB, xA:xB]
                if cell.size == 0:
                    continue
                if return_geometry:
                    cells_out.append((cid, r, c, cell, (xA, yA, xB, yB)))
                else:
                    cells_out.append((cid, r, c, cell))
    return cells_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for sh in args.sheets:
        base = os.path.basename(sh)[:-4]
        img = cv2.imread(sh)
        if img is None:
            print(f"SKIP {sh}: unreadable")
            continue
        print(f"{base}:")
        for cid, r, c, cell in sheet_to_cells(img, verbose=True):
            name = f"{base}_card{cid}_r{r}c{c}.jpg"
            cv2.imwrite(os.path.join(args.out, name), cell)
            total += 1
    print(f"DONE {total} cells -> {args.out}")


if __name__ == "__main__":
    main()