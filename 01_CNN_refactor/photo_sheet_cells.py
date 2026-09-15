import sys
"""Phone-photo sheet -> cells, using the server pipeline's Hough geometry.

The scan-tuned sheet_to_cells assumes flatbed-bright paper (fixed <190
threshold); phone photos are darker/noisier, so its thin grid line detector
floods and rows drift. This variant keeps the hue-based teal band detector
for card tops but fits each card's grid with the server pipeline's
Hough-based line detection (pipeline.find_grid_line_positions) + equal-step
boundary snapping, which tolerate lighting gradients and noisy cells.

Caller should pass an already-normalized (brightened) BGR image, e.g. via
evaluate_photo_sheets.normalize.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import _cell_boundaries, find_grid_line_positions  # noqa: E402
from extract_scan_cells import (  # noqa: E402
    INSET,
    _BAND_FAMILIES,
    _header_hue_spec,
    _hue_in,
    find_header_bands,
)


def sheet_to_cells_photo(img, verbose=False, return_geometry=False):
    """Return list of (card_id, row, col, cell) with optional geometry,
    mirroring extract_scan_cells.sheet_to_cells's contract."""
    bands = find_header_bands(img)
    if not bands:
        return []
    H, W = img.shape[:2]
    cells_out = []
    for i, (ts, te) in enumerate(bands):
        top = te + 6
        bottom = bands[i + 1][0] if i + 1 < len(bands) else H
        if bottom - top < 60:
            if verbose:
                print(f"card{i + 1}: region too thin ({bottom - top}px)")
            continue
        region = img[top:bottom, :]
        hy, vx = find_grid_line_positions(region)
        if verbose:
            print(f"card{i + 1}: region ({top}..{bottom}) hlines={hy} vlines={vx}")
        if len(hy) < 4 or len(vx) < 4:
            if verbose:
                print(f"card{i + 1}: not enough lines (h={len(hy)} v={len(vx)})")
            continue
        rb = _cell_boundaries(hy, region.shape[0])
        cb = _cell_boundaries(vx, region.shape[1])
        if verbose:
            print(f"card{i + 1}: row bounds {rb}, col bounds {cb}")
        for r in range(5):
            for c in range(5):
                y0, y1 = top + rb[r], top + rb[r + 1]
                x0, x1 = cb[c], cb[c + 1]
                if y1 - y0 <= 2 * INSET or x1 - x0 <= 2 * INSET:
                    continue
                cell = img[y0 + INSET:y1 - INSET, x0 + INSET:x1 - INSET]
                if cell.size == 0:
                    continue
                if return_geometry:
                    cells_out.append((i + 1, r, c, cell,
                                      (x0, y0, x1, y1)))
                else:
                    cells_out.append((i + 1, r, c, cell))
    return cells_out


def teal_card_bboxes(img):
    """Return [(x0, y0, x1, y1)] per card header band, merged across hue
    splits, sorted by y.  Works for any sheet print color (the header
    color is auto-detected, see _header_hue_spec)."""
    spec = _header_hue_spec(img)
    if spec is None:
        return []
    hl, hh, s_min, v_min, rowfrac = spec
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int)
    s = hsv[:, :, 1].astype(int)
    v = hsv[:, :, 2].astype(int)
    teal = _hue_in(h, hl, hh) & (s > s_min) & (v > v_min)
    rowcnt = teal.mean(axis=1)
    rows = [y for y in range(len(rowcnt)) if rowcnt[y] > rowfrac]
    runs = []
    for y in rows:
        if runs and y - runs[-1][-1] <= 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    runs = [r for r in runs if len(r) >= 15]
    merged = []
    for r in runs:
        if (merged and r[0] - merged[-1][1] < 120
                and cv2.countNonZero(teal[merged[-1][1]:r[0], :]) == 0):
            merged[-1][1] = r[-1]
        else:
            merged.append([r[0], r[-1]])
    boxes = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Frame-level paper extent (hard constraint - paper can't be wider than this)
    frame_ext = _frame_bright_extent(gray)
    for y0, y1 in merged:
        cols = np.where(teal[y0:y1 + 1, :].any(axis=0))[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])

        # Clamp horizontal extent to the paper's bright-paper region.
        # The teal header color can bleed/reflect onto the desk, making
        # the mask wider than the actual card. Use the card body region
        # (just below the header) to find the true paper edges.
        body_y0 = min(y1 + 2, gray.shape[0] - 1)
        body_y1 = min(y1 + 60, gray.shape[0])
        if body_y1 > body_y0:
            body_band = gray[body_y0:body_y1, :]
            colmean = body_band.mean(axis=0).astype(np.uint8)
            thr, _ = cv2.threshold(colmean, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright = colmean > max(40, int(thr) * 0.90)
            xs = np.where(bright)[0]
            if xs.size > 0:
                x0 = max(x0, int(xs[0]))
                x1 = min(x1, int(xs[-1]))

        # Also clamp to frame-level paper extent as a hard constraint.
        # The header mask can bleed onto the desk; the full-frame paper
        # detection is more reliable for the absolute outer bounds.
        if frame_ext is not None:
            fx0, fx1 = frame_ext
            x0 = max(x0, fx0)
            x1 = min(x1, fx1)

        boxes.append((x0, y0, x1, y1))
    boxes = sorted(boxes, key=lambda b: b[1])
    if len(boxes) == 1 and gray.shape[0] >= 1500:
        extra = _faint_header_fallback(teal, boxes[0])
        if extra:
            boxes = sorted(extra, key=lambda b: b[1])
    return boxes


def _faint_header_fallback(teal, first_box):
    """Recover the remaining card headers when a full strip registers only
    one band.  Faint/narrow header prints still peak strongly INSIDE their
    own x-window even when they lose the full-width per-row fraction, so
    scan the detected band's columns for the top-3 strongly-separated teal
    rows and rebuild one box per peak (uniform pitch check included)."""
    x0, _, x1, _ = first_box
    sub = teal[:, x0:x1].mean(axis=1).astype(np.float64)
    sub = np.convolve(sub, np.ones(7) / 7, mode="same")
    best = sub.max()
    if best <= 0.01:
        return []
    peaks = []
    for y in range(2, sub.size - 2):
        if (sub[y] > 0.35 * best and sub[y] >= sub[y - 2]
                and sub[y] >= sub[y - 1] and sub[y] >= sub[y + 1]
                and sub[y] >= sub[y + 2]):
            peaks.append((y, sub[y]))
    peaks.sort(key=lambda p: -p[1])
    kept = []
    for y, _ in peaks:
        if all(abs(y - k) >= 250 for k in kept):
            kept.append(y)
        if len(kept) == 3:
            break
    if len(kept) != 3:
        return []
    kept.sort()
    pitches = [kept[i + 1] - kept[i] for i in range(2)]
    if max(pitches) - min(pitches) > 0.28 * max(pitches):
        return []
    boxes = []
    for pk in kept:
        lo, hi = pk, pk
        while lo > 2 and sub[lo - 1] >= 0.5 * sub[pk]:
            lo -= 1
        while hi < sub.size - 3 and sub[hi + 1] >= 0.5 * sub[pk]:
            hi += 1
        boxes.append((x0, lo, x1, hi))
    return boxes


def _snap(bounds, dark, lo, hi, radius):
    """Move each boundary to the darkest row/col within radius of it,
    clamped to [lo + 2, hi - 2]. dark: array of darkness per line."""
    out = list(bounds)
    for i in range(1, len(out) - 1):
        a, b = max(lo + 2, out[i] - radius), min(hi - 2, out[i] + radius)
        if b < a:
            continue
        out[i] = a + int(np.argmax(dark[a:b + 1]))
    return out


def _card_y_extent(gray, top, hint_bottom):
    """Clip the card's bottom to the last bright (paper) row in the region,
    so the last card doesn't drag cells into the dark background. Row
    MEDIAN resists bright specks in the dark backdrop that defeat the mean."""
    reg = gray[top:hint_bottom, :].astype(np.float32)
    if reg.size == 0:
        return hint_bottom
    rowmed = np.median(reg, axis=1)
    hi = np.percentile(rowmed, 90)
    thr = max(40, hi - 35)
    below = np.where(rowmed > thr)[0]
    if below.size == 0:
        return hint_bottom
    return min(hint_bottom, top + int(below[-1]) + 2)


def _grid_dip(gray, x0, x1, top, bottom, strip_w=12):
    """Per-row brightness-dip score in a narrow vertical strip centred on
    the card's column extent.  Printed grid lines are thin dark horizontal
    rows that are much darker than their neighbours; a full-width mean
    dilutes them into noise, but a narrow strip preserves the signal."""
    H = gray.shape[0]
    xm = (x0 + x1) // 2
    xL = max(0, xm - strip_w // 2)
    xR = min(gray.shape[1], xm + strip_w // 2)
    strip = gray[top:bottom, xL:xR].mean(axis=1).astype(np.float64)
    n = len(strip)
    score = np.zeros(n)
    for j in range(5, n - 5):
        neighbours = np.concatenate([strip[j - 5:j - 1], strip[j + 2:j + 6]])
        score[j] = neighbours.mean() - strip[j]
    full = np.zeros(H)
    full[top:top + n] = score
    return full


def _grid_dip_v(gray, x0, x1, top, bottom, strip_h=None):
    """Per-column brightness-dip score across the full card height.
    Vertical grid lines are continuous dark lines spanning the card;
    digit strokes are localized to rows. Averaging over full height
    suppresses digit strokes while preserving grid lines."""
    W = gray.shape[1]
    if strip_h is None:
        yL, yR = top, bottom  # full card height
    else:
        ym = (top + bottom) // 2
        yL = max(0, ym - strip_h // 2)
        yR = min(gray.shape[0], ym + strip_h // 2)
    strip = gray[yL:yR, x0:x1].mean(axis=0).astype(np.float64)
    n = len(strip)
    score = np.zeros(n)
    for j in range(5, n - 5):
        neighbours = np.concatenate([strip[j - 5:j - 1], strip[j + 2:j + 6]])
        score[j] = neighbours.mean() - strip[j]
    full = np.zeros(W)
    full[x0:x0 + n] = score
    return full


def _find_grid_lines_phone(card_img, verbose=False):
    """Phone-photo variant of find_grid_line_positions with relaxed params.
    Phone photos have fainter grid lines; lower thresholds and shorter min length.
    """
    from pipeline import _cluster_1d, _select_best_lines
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # Lower Canny thresholds for fainter lines
    edges = cv2.Canny(gray, 30, 100)
    h, w = edges.shape

    # More permissive: shorter min length (8% vs 15%), lower Hough threshold
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30,
        minLineLength=int(w * 0.08), maxLineGap=50,
    )

    horiz_ys, vert_xs = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5
            if abs(dy) <= 5 and length >= w * 0.08:
                horiz_ys.append((y1 + y2) // 2)
            elif abs(dx) <= 5 and length >= h * 0.08:
                vert_xs.append((x1 + x2) // 2)

    horiz_clusters = _cluster_1d(horiz_ys, tol=h * 0.02)
    vert_clusters = _cluster_1d(vert_xs, tol=w * 0.02)

    h_margin = h * 0.03
    w_margin = w * 0.03
    horiz_clusters = [(y, c) for y, c in horiz_clusters if h_margin < y < h - h_margin]
    vert_clusters = [(x, c) for x, c in vert_clusters if w_margin < x < w - w_margin]

    horiz_ys = _select_best_lines(horiz_clusters)
    vert_xs = _select_best_lines(vert_clusters)
    if verbose:
        print(f"  phone Hough: h={len(horiz_ys)} v={len(vert_xs)} lines")
    return horiz_ys, vert_xs


def _fix_row_lattice(rb, dip, top, bottom, gray=None, x0=0, x1=0):
    """Rebuild a uniform 6-boundary row lattice from the snapped interior
    dividers, tolerating missing and mis-snapped dividers:
      - a gap near 2*pitch with a dip peak at the midpoint means the divider
        was skipped: fill it in (confirmed by the dip);
      - dividers that don't fit the resulting pitch are outliers: drop them.
    The top/bottom borders are extrapolated from the interior pitch, so a
    guessy region bottom (e.g. the image edge under the last card) can't
    stretch the rows.
    When gray is supplied, the extrapolated TOP border is sanity-checked
    against the printed card: if the cell it would carve out (rows
    kept[0]-pitch .. kept[0]) is a solid graphic block instead of a digit
    row (e.g. a QR/promo plaque printed between the header band and the
    grid), the whole lattice is anchored one slot too high and every cell
    slices into the row below.  Detect it and re-anchor the lattice AT the
    first proven interior line, so cells align with the printed rows."""

    def _top_blank_ratio(y0, y1):
        """Fraction of rows in [y0,y1) that are near-blank across the card's
        width.  Digit rows have printed strokes on paper so a noticeable
        share of rows stay nearly clean; a solid graphic block (QR/promo
        plaque) covers every row."""
        if y1 - y0 < 2 or x1 <= x0:
            return None
        row = (gray[y0:y1, x0:x1] < 120).mean(axis=1)
        return float((row < 0.10).mean()), float(row.mean())
    if len(rb) != 6:
        return rb
    divs = sorted(rb[1:5])
    pitch = float(np.median(np.diff(np.array(divs, float)))) if len(divs) >= 2 else 0.0

    # 1) fill missing dividers (gap ≈ 2*pitch) confirmed by the dip
    kept = []
    for j in range(len(divs) - 1):
        g = divs[j + 1] - divs[j]
        kept.append(divs[j])
        if pitch > 0 and 1.8 * pitch <= g <= 2.35 * pitch:
            mid = divs[j] + int(round(pitch))
            a, b = max(top + 2, mid - 8), min(bottom - 2, mid + 8)
            a, b = int(a), int(b)
            if b > a:
                kept.append(a + int(np.argmax(dip[a:b + 1])))

    # 2) drop dividers that don't fit the pitch (iterate on worst fit)
    while len(kept) >= 4:
        arr = np.array(kept, float)
        gaps = np.diff(arr)
        p = float(np.median(gaps))
        if np.max(np.abs(gaps - p)) <= 0.12 * p:
            break
        impact = []
        for j in range(len(kept)):
            gg = list(gaps)
            if 0 < j < len(kept) - 1:
                gg[j - 1] = arr[j + 1] - arr[j - 1]
                gg.pop(j)
            else:
                gg.pop(0 if j == 0 else -1)
            gg = np.array(gg, float)
            pj = float(np.median(gg)) if gg.size else p
            impact.append(float(np.max(np.abs(gg - pj))) if gg.size else 0.0)
        j = int(np.argmin(impact))
        kept.pop(j)

    if len(kept) >= 3:
        arr = np.array(kept, float)
        p = float(np.median(np.diff(arr)))
        p = max(70.0, min(120.0, p))
        base = kept[0] - p
        rb = []
        for k in range(6):
            v = base + k * p
            interior = 0 < k < 5
            far = abs(v - top) > p and abs(v - bottom) > p
            if interior or far:
                a, b = max(top + 2, int(v) - 6), min(bottom - 2, int(v) + 6)
                if b > a:
                    v = float(a + int(np.argmax(dip[a:b + 1])))
            rb.append(int(round(v)))
        if gray is not None and x1 > x0 and len(rb) == 6 and rb[1] - rb[0] >= 40:
            ta, tb = max(top + 1, rb[0] - 8), min(bottom - 1, rb[0] + 8)
            top_line = float(np.max(dip[ta:tb + 1])) if tb > ta else 0.0
            top_anchor = float(dip[rb[1]]) if 0 <= rb[1] < dip.size else 0.0
            top_b, top_d = _top_blank_ratio(rb[0], rb[1])
            below_b, _ = _top_blank_ratio(rb[1], rb[2])
            # A QR/promo plaque printed between the header band and the grid
            # covers every row with ink (no near-blank rows) and is packed
            # (>40% dark).  Real digit rows - even a first row that overhangs
            # the band edge - always leave a band of blank rows (>= ~6% of
            # the cell) and lower ink density, and cards whose top boundary
            # merely overhangs the header band (blank ratio just below the
            # row beneath, e.g. card 2) stay anchored as-is.
            if (top_b is not None and below_b is not None and
                    top_b < 0.02 and top_d > 0.40 and below_b > 0.03 and
                    top_b < 0.5 * below_b and
                    top_line < 60 and top_anchor >= 0.5 * p):
                # kept[0] is the grid's real top line: anchor the lattice
                # there.  The last row extends past the card bottom (a
                # truncated/blank row, flagged if unreadable).
                rb = [rb[1] + int(round(k * p)) for k in range(6)]
                for k in range(1, 5):
                    a, b = max(top + 2, rb[k] - 6), min(bottom - 2, rb[k] + 6)
                    if b > a:
                        rb[k] = a + int(np.argmax(dip[a:b + 1]))
        return rb

    # fallback: old uniform lattice through all 6 boundaries
    idx = np.array([0, 1, 2, 3, 4, 5], float)
    a, b = np.polyfit(idx, np.array(rb, float), 1)
    pitch = max(70.0, min(120.0, a))
    return [int(round(b + pitch * k)) for k in range(6)]


_header_templates_cache = None


def _get_header_templates():
    """Lazy-load the BINGO header templates for column homography."""
    global _header_templates_cache
    if _header_templates_cache is None:
        try:
            from bingo_header_locator import load_template_features
        except ImportError:
            _header_templates_cache = []
            return _header_templates_cache
        tdir = Path(__file__).parent / "header_templates"
        if not tdir.exists():
            _header_templates_cache = []
            return _header_templates_cache
        paths = [str(p) for p in sorted(tdir.iterdir()) if p.is_file()]
        detector = cv2.SIFT_create()
        _header_templates_cache = load_template_features(detector, paths)
    return _header_templates_cache


def _header_homography_columns(gray, boxes, grid_x0, grid_x1):
    """Per-card 6-boundary column lattice from the header-template SIFT
    homography.

    The BINGO header letters print strongly even on sheets whose number-grid
    ink is washed out, so SIFT features on them recover a robust homography.
    A card is matched on its OWN tight header band (full-sheet matching
    fragments the keypoints across every card and starves the weaker cards),
    each match is validated against sane geometry, and any card that can't be
    matched (e.g. a dark/decorative header variant with no template) inherits
    the nearest validated card's columns shifted by its header-box x0 offset
    -- the sheet's printed columns run through all cards, so that offset
    reconstructs them to within a few px, far better than equal division over
    a bright-extent whose right edge sits on a desk.  Returns a list (one
    entry per header box) of [x0, B|I, I|N, N|G, G|O, x5] or None when no
    card on the sheet could be validated (caller keeps equal division)."""
    if not boxes:
        return []
    try:
        import bingo_header_locator as hdr
    except ImportError:
        return [None] * len(boxes)
    templates = _get_header_templates()
    if not templates:
        return [None] * len(boxes)

    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    centers = [(y0 + y1) / 2 for (_, y0, _, y1) in boxes]
    x0s = [x0 for (x0, _, _, _) in boxes]
    out = [None] * len(boxes)

    # Match each card against its own header band so keypoints aren't shared
    # out with the other cards' matches.  Every template is tried on the full,
    # unconsumed band (detect_all_instances' keypoint-consumption loop lets a
    # high-inlier DEGENERATE match -- a collapsed homography with pitch ~ 0 --
    # starve the correct full-grid match, so we score all candidates instead
    # and keep only geometrically sane ones).
    cands_all = [[] for _ in boxes]
    for i, (bx0, by0, bx1, by1) in enumerate(boxes):
        mx = max(12, int(0.05 * (bx1 - bx0)))
        my = max(12, int(0.05 * (by1 - by0)))
        wl = max(0, bx0 - mx)
        wt = max(0, by0 - my)
        band = gray[wt:by1 + my, wl:bx1 + mx]
        if band.shape[0] < 20 or band.shape[1] < 20:
            continue
        skp, sdes = detector.detectAndCompute(band, None)
        if sdes is None or len(skp) < 10:
            continue
        active = np.ones(len(skp), dtype=bool)
        for templ in templates:
            res = hdr.find_best_instance(matcher, templ, skp, sdes, active)
            if res is None:
                continue
            proj = hdr.project_boundaries({"H": res["H"], "template": templ})
            band_y = proj["B|I"]["top"][1] + wt
            holes = [float(proj[k]["bottom"][0]) + wl
                     for k in ("B|I", "I|N", "N|G", "G|O")]
            pitch = float(np.median(np.diff(np.array(holes, float))))
            if not (85.0 < pitch < 115.0):
                continue
            if abs(band_y - centers[i]) > 150:
                continue
            if not (holes[0] > grid_x0 and holes[-1] < grid_x1):
                continue
            cands_all[i].append((res["num_inliers"], pitch, holes))

    # Sheet-level pitch consensus: the same printed grid joins every card, so
    # all cards share one pitch.  Reject outliers vs the median before use.
    all_pitch = [p for c in cands_all for (_, p, _) in c]
    med_pitch = float(np.median(all_pitch)) if all_pitch else 0.0
    for i, cands in enumerate(cands_all):
        keep = [(n, p, h) for (n, p, h) in cands
                if med_pitch == 0.0 or abs(p - med_pitch) <= 0.12 * med_pitch]
        if not keep:
            continue
        n, p, holes = max(keep, key=lambda t: t[0])
        out[i] = ([max(0, int(round(holes[0] - p)))]
                  + [int(round(x)) for x in holes]
                  + [int(round(holes[-1] + p))])

    # Cards the projection couldn't reach (dark/foreign header, occluded,
    # extreme skew): inherit the nearest validated card's columns, shifted by
    # that card's header-box x0 offset (the printed columns pass through all
    # cards of the sheet).  pitch and spacing are preserved.  A stacked
    # sheet's card headers sit only a few px apart, so a large offset means
    # the header box is garbage (e.g. a washed-out forced scan) — inherit
    # unshifted rather than shifting the whole grid off the paper.
    matched = [k for k, c in enumerate(out) if c is not None]
    if matched:
        for i, cols in enumerate(out):
            if cols is not None:
                continue
            src = min(matched, key=lambda s: abs(centers[i] - centers[s]))
            gaps = np.diff(np.array(out[src], float))
            pitch = float(np.median(gaps)) if gaps.size else 100.0
            shift = x0s[i] - x0s[src]
            if abs(shift) > 0.35 * pitch:
                shift = 0
            out[i] = [x + shift for x in out[src]]
    return out


def _sheet_strong_clusters(boxes, gray):
    """Cross-card vertical line clusters with support >= 2.  Every printed
    column divider spans all cards of the sheet, so only grid lines (or
    full-height desk/paper artifacts) get multi-card support; digit strokes
    are bounded per cell and scatter."""
    cand = []
    H = gray.shape[0]
    for i, b in enumerate(boxes):
        top = b[1] + 2
        bottom = boxes[i + 1][1] if i + 1 < len(boxes) else H
        if bottom - top < 60:
            continue
        _, vx = find_grid_line_positions(
            cv2.cvtColor(gray[top:bottom, :], cv2.COLOR_GRAY2BGR))
        cand.extend(vx)
    cand = np.sort(np.array(cand, float))
    runs = []
    for x in cand:
        if runs and abs(x - runs[-1][-1]) <= 10.0:
            runs[-1].append(x)
        else:
            runs.append([x])
    return [(float(np.mean(c)), len(c)) for c in runs if len(c) >= 2]


def _best_column_lattice(strong, min_header_x0):
    """Rebuild the sheet's uniform 6-column lattice from strong cross-card
    vertical-line clusters.

    Equal division over the bright-paper extent assumes the extent's edges
    ARE the printed grid's outer borders.  When shadows/folds move one of
    those edges, every column drifts and cells slice digits.  The printed
    grid lines themselves are the robust reference: they run through every
    card, and the card's header left edge (the sheet's printed left border)
    anchors the phase.  Pitch comes from the high-support cluster gaps;
    phase is chosen so a boundary sits nearest the header left edge."""
    if len(strong) < 2:
        return None
    xs = np.sort(np.array([float(x) for x, _ in strong]))
    gaps = np.diff(xs)
    ok = gaps[(gaps >= 85.0) & (gaps <= 125.0)]
    if ok.size == 0:
        return None
    p = int(round(float(np.median(ok))))
    best = None
    for x in xs:
        x0s = [x - k * p for k in range(-2, 3)
               if min_header_x0 - 45 <= x - k * p <= min_header_x0 + 45]
        if not x0s:
            continue
        x0 = min(x0s, key=lambda v: abs(v - min_header_x0))
        if abs(x0 - min_header_x0) > 20:
            continue
        slots = np.round((xs - x0) / p)
        dist = np.abs(xs - (x0 + p * slots))
        okk = (dist <= 5.0) & (slots >= 0) & (slots <= 5)
        n_in = int(okk.sum())
        n_slots = len({int(s) for s, d in zip(slots, dist) if d <= 5.0 and 0 <= s <= 5})
        if (best is None or n_in > best[0]
                or (n_in == best[0] and n_slots > best[2])
                or (n_in == best[0] and n_slots == best[2]
                    and abs(x0 - min_header_x0) < best[3])):
            best = (n_in, x0, n_slots, abs(x0 - min_header_x0))
    if best is None or best[0] < 2:
        return None
    # Trust the lattice only when it actually explains MOST of the
    # strong clusters.  Spurious cross-card lines (digit-column shading,
    # desk artifacts) scatter; a real printed grid puts every strong
    # line on slots of one uniform lattice.
    if best[0] < max(2, int(np.ceil(0.8 * len(xs)))):
        return None
    return int(round(best[1])), p


def _sheet_to_cells_with_boxes(
        img, boxes, verbose=False, return_geometry=False, trace=None):
    """Shared per-card cell extraction.  Rows use the proven brightness-dip
    + snap (robust on normalized phone images).  Columns use pipeline Hough
    on the full card width (from frame bright extent) + uniform lattice fit.

    trace: optional dict that receives the geometry DECISIONS made here
    (boxes, extent, clusters/lattice, per-card rows/cols and whether the
    homography or lattice path was used) so a misread can be diagnosed
    from the degraded /scan-card debug payload or server log."""
    if not boxes:
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get frame-level paper extent once from grid regions below headers.
    # This gives the true horizontal bounds of the card grid.
    frame_ext = _frame_bright_extent(gray, boxes)
    if frame_ext is not None:
        grid_x0, grid_x1 = frame_ext
    else:
        grid_x0, grid_x1 = 0, W

    # Column fallback: per-card header-template homography.  The BINGO
    # header letters print strongly even on sheets whose number-grid ink is
    # faint, so a confident SIFT match there yields perspective-accurate
    # column boundaries.  Equal division over the shared bright extent is
    # fragile when the paper/desk runs past the printed grid's right border:
    # it inflates the pitch and drifts columns off the grid (over-wide cells,
    # off-page right edge).  When SIFT matches a card we trust the projected
    # columns; otherwise cards keep the proven equal-division + dip path.
    hom_cols = None
    try:
        hom_cols = _header_homography_columns(gray, boxes, grid_x0, grid_x1)
    except Exception:
        hom_cols = None

    # Printed-grid reference: cross-card vertical line clusters.  Every
    # printed column divider spans all cards, so clusters supported by >=2
    # cards locate the true grid even when the bright-paper extent's edges
    # float off it (shadowed/folded sheet side).
    strong = []
    lattice = None
    try:
        strong = _sheet_strong_clusters(boxes, gray)
        if len(strong) >= 2:
            lo_b = min(b[0] for b in boxes)
            lattice = _best_column_lattice(strong, lo_b)
    except Exception:
        strong = []
    # Cards of one printed sheet share the same horizontal layout; when the
    # teal header strips are x-aligned the grid positions are identical on
    # every card, so the sheet lattice evidence extends to all of them.
    sheet_aligned = all(abs(b[0] - min(b[0] for b in boxes)) <= 15
                        for b in boxes)

# When cards are not x-aligned, the header boxes may have different
    # widths (e.g. one card's header detection includes table to the right).
    # The LEFT edge of the header is typically at the true card paper edge,
    # while the right edge may extend onto the table.  However, individual
    # header detections can have outlier left edges (shadows, reflections).
    # Use the MEDIAN left edge of all headers as the consistent left edge
    # for all cards, with a fixed TARGET_CARD_WIDTH.
    TARGET_CARD_WIDTH = 475
    left_edges = [b[0] for b in boxes]
    median_left = int(np.median(left_edges))
    card_exts = [(median_left, median_left + TARGET_CARD_WIDTH) for _ in boxes]

    # For aligned sheets, still use shared frame extent (already computed)
    # but for unaligned, use normalized per-card extents above.

    cells_out = []
    prev_pitch = None
    prev_pitch2 = None
    for i, (hx0, y0, hx1, y1) in enumerate(boxes):
        top = y1 + 2
        bottom = boxes[i + 1][1] if i + 1 < len(boxes) else H
        if i + 1 == len(boxes):
            bottom = _card_y_extent(gray, top, bottom)
        if bottom - top < 60 or grid_x1 - grid_x0 < 60:
            if verbose:
                print(f"card{i + 1}: region too thin ({bottom - top}x{grid_x1 - grid_x0})")
            continue
        
        # Use header box for vertical position (y), but frame extent for horizontal (x)
        # Use per-card extent when cards are not x-aligned;
        # otherwise use shared frame extent.
        if sheet_aligned:
            x0, x1 = grid_x0, grid_x1
        else:
            x0, x1 = card_exts[i]

# ---- ROWS: proven dip + snap ----
        grid_sig = _grid_dip(gray, x0, x1, top, bottom)
        # For the last card the region bottom (paper edge) can be well below
        # the printed card bottom, inflating the equal-division pitch and
        # seeding the snap mid-stroke even when the grid lines are faint.
        # Seed rows from the previous card's solved pitch instead, which is a
        # stable prior (perspective drift card-to-card is a few px).
        if i + 1 == len(boxes):
            if prev_pitch is not None and prev_pitch2 is not None:
                # Perspective: each lower card is closer to the camera, so its
                # printed grid is *larger* in image space (blue sheet scan
                # 1789266911: card pitches 70 -> 77 -> ~89).  Copying the
                # previous card's pitch shortchanges the last card — rows get
                # compressed and sliced into the row above, and the last row is
                # dropped entirely.  Extrapolate the growth trend instead:
                # near-frontal photos have pitch ~ proportional card-over-card.
                seed_pitch = prev_pitch * prev_pitch / max(1.0, prev_pitch2)
                seed_pitch = max(50.0, min(140.0, seed_pitch))
                rb = [top + round(seed_pitch * k) for k in range(6)]
            elif prev_pitch is not None:
                rb = [top + round(prev_pitch * k) for k in range(6)]
            else:
                rb = [top + (bottom - top) * k // 5 for k in range(6)]
        else:
            rb = [top + (bottom - top) * k // 5 for k in range(6)]
        rb = _snap(rb, grid_sig, top, bottom, radius=40)

        # ---- COLUMNS: equal division with tight snapping to detected peaks ----
        # Use equal division as the strong prior. Only snap to detected peaks
        # if they are very close to the expected positions. This avoids the
        # sequential-snapping error propagation when dividers are missing.
        card_crop = img[top:bottom, x0:x1]
        _, vert_xs = find_grid_line_positions(card_crop)
        width = x1 - x0

        cb_rel = [round(width * k / 5) for k in range(6)]
        tol = max(10, int(0.12 * width / 5))
        for k in range(1, 5):
            expected = round(width * k / 5)
            candidates = [p for p in vert_xs if abs(p - expected) <= tol]
            if candidates:
                cb_rel[k] = min(candidates, key=lambda p: abs(p - expected))
        cb = [x0 + x for x in cb_rel]
        cb_eq = list(cb)
        vx_abs = [x0 + h for h in vert_xs]

        # ---- COLUMN PITCH: correct per-card pitch from the vertical dip ----
        # The shared extent (x0..x1) over-estimates pitch when the sheet is
        # slightly rotated, slicing the left edge of column 5.  Snap each
        # interior boundary to the STRONGEST vertical-grid dip in a window,
        # rebuild the uniform lattice from that interior pitch (borders
        # extrapolated), and keep the result only if it actually sits ON the
        # dip lines better than the equal-division hough-snap — so sheets
        # whose pitch or noise don't fit this correction keep the proven
        # hough-based columns.
        dv = _grid_dip_v(gray, max(0, x0 - 90),
                         min(gray.shape[1] - 1, x1 + 90), top, bottom)
        # The equal-division/hough columns assume a perfectly uniform lattice
        # over the shared extent.  When the sheet is slightly rotated the true
        # per-card pitch differs, so the interior dividers become uneven.
        # Rebuild a per-card lattice from the STRONGEST vertical dips, but only
        # trust it when its dividers coincide with LONG vertical Hough lines
        # (printed grid dividers) at least as well as the equal division does.
        # Short digit strokes yield strong dips but no long-line support, so
        # they can't hijack the lattice on sheets whose grid is uniform.
        eq = [x0 + round(width * k / 5) for k in range(6)]
        r = int(0.22 * width / 5)
        snapped = list(eq)
        for k in range(1, 5):
            a, b = max(0, snapped[k] - r), min(gray.shape[1] - 1, snapped[k] + r)
            if b > a:
                snapped[k] = a + int(np.argmax(dv[a:b + 1]))
        cand = _fix_row_lattice(np.array(snapped, float), dv, 0,
                                gray.shape[1] - 1)

        def _hough_agree(lattice):
            """Count of interior dividers that coincide with a LONG vertical
            Hough line (the printed grid dividers; short digit strokes won't
            accumulate enough votes to appear)."""
            n = 0
            for k in range(1, 5):
                if any(abs(lattice[k] - (x0 + h)) <= 3 for h in vert_xs):
                    n += 1
            return n

        if (_hough_agree(cand) >= 2 and _hough_agree(cand) >= _hough_agree(cb) and
                min(cand) >= x0 - 25 and max(cand) <= x1 + 25 and
                sum(dv[cand[k]] for k in range(1, 5)) >
                sum(dv[cb[k]] for k in range(1, 5))):
            cb = cand

        # Pitch sanity: if BOTH hom unavailable AND cand failed (cb == cb_eq),
        # the equal-division extent is likely wrong. Narrow/widen to ~95px pitch.
        hom_available = hom_cols is not None and hom_cols[i] is not None
        if not hom_available and cb == cb_eq:
            eq_pitch = (cb_eq[-1] - cb_eq[0]) / 5.0
            if eq_pitch > 130:
                target_width = int(95.0 * 5)
                cur_width = cb_eq[-1] - cb_eq[0]
                excess = cur_width - target_width
                card_x0 = cb_eq[0] + excess // 2
                card_x1 = card_x0 + target_width
                width = card_x1 - card_x0
                cb_eq = [card_x0 + round(width * k / 5) for k in range(6)]
                cb = list(cb_eq)
            elif eq_pitch < 70:
                target_width = int(95.0 * 5)
                cur_width = cb_eq[-1] - cb_eq[0]
                if cb_eq[0] > 0 and cb_eq[-1] < gray.shape[1]:
                    excess = target_width - (cb_eq[-1] - cb_eq[0])
                    card_x0 = max(0, cb_eq[0] - excess // 2)
                    card_x1 = card_x0 + target_width
                    width = card_x1 - card_x0
                    cb_eq = [card_x0 + round(width * k / 5) for k in range(6)]
                    cb = list(cb_eq)

        # Homography is used ONLY when it disagrees with the equal-division+dip
        # columns just computed -- i.e. when the printed grid is faint and the
        # dip/Hough lattice floats off it (B 121->97, C 142->101, S2 124->97).
        # A >5% pitch disagreement is enough: the orange sheet 1789240382
        # showed a modest ~7% float (equal division 107 vs true ~100) that
        # sliced into digits and ran off the sheet's right edge, below the
        # old 10% bar.  When the two pitches agree, equal division already
        # recovered the true lattice and the header match would merely
        # re-draw the same grid shifted a few px, so we keep the proven
        # equal-division columns (approved sheets stay byte-stable).
        use_hom = False
        use_hom = False
        if hom_cols is not None and hom_cols[i] is not None:
            h_gaps = np.diff(np.array(hom_cols[i][1:5], float))
            h_pitch = float(np.median(h_gaps)) if h_gaps.size else 0.0
            cb_gaps = np.diff(np.array(cb[1:5], float))
            cb_pitch = float(np.median(cb_gaps)) if cb_gaps.size else 0.0
            use_hom = (cb_pitch > 0.0 and abs(h_pitch - cb_pitch) >
                       0.05 * cb_pitch)
        if use_hom:
            # Card's header matched a template: the BINGO letters anchor a
            # perspective-accurate homography, so its projected columns beat
            # the faint-grid equal division (which floats off the printed
            # grid / bright desk).  But on some sheets the template match
            # projects a grid that is the wrong scale (e.g. pink sheet
            # 1789340606 card1 where hom spans only 56% of the fallback
            # lattice width).  Reject hom if its column span is too small
            # relative to the fallback lattice, unless the fallback itself
            # has an unreasonable pitch (in which case hom may be the only
            # usable signal, as with yellow sheets).
            hc = hom_cols[i]

            # Span check: compare hom span to FALLBACK lattice span (cb
            # before hom adoption).  A correct homography projects the
            # full template grid; a mismatched one often projects only a
            # subset.  But if the fallback span is unreasonably large
            # (>600px, likely the shared sheet extent rather than per-card),
            # the fallback is unreliable and we should be lenient with hom.
            fallback_span = cb[-1] - cb[0]
            hom_span = hc[-1] - hc[0]
            # Expected single-card span ~5 * 95 = 475px
            if fallback_span > 600:
                # Fallback is likely the shared extent; use lenient threshold
                threshold = 0.5
            else:
                threshold = 0.7
            if hom_span < threshold * fallback_span:
                use_hom = False
                # Fall through to keep cb

            # Position check: hom columns must overlap with the card's
            # horizontal extent (x0..x1). If hom projects completely
            # outside the card's extent, it's matching something else
            # (table edge, reflection) and should be rejected.
            if use_hom:
                if hom_span > 0:
                    # Check if any hom column falls within the card extent
                    overlap = any(x0 <= x <= x1 for x in hc)
                    if not overlap:
                        use_hom = False
                        # Fall through to keep cb

            def _margin_ink(cols, rows):
                total = 0
                for r in range(len(rows) - 1):
                    y0, y1 = rows[r], rows[r + 1]
                    for c in range(1, 5):
                        x = cols[c]
                        if x - 7 >= 0:
                            total += int((gray[y0:y1, x - 7:x] < 180).sum())
                        if x + 7 <= gray.shape[1]:
                            total += int((gray[y0:y1, x:x + 7] < 180).sum())
                return total

            if use_hom:
                if cb == cb_eq:
                    # No corrected lattice available (cand failed).
                    # Homography is the best signal we have; adopt it.
                    cb = hc
                else:
                    # A corrected lattice exists (cb != cb_eq).
                    # The corrected lattice was validated by Hough lines;
                    # only adopt hom if it's CLEARLY better (>10% less ink).
                    ink_hc = _margin_ink(hc, rb)
                    ink_cb = _margin_ink(cb, rb)
                    if ink_hc < ink_cb * 0.9:
                        cb = hc

        # ---- COLUMN REWIRE: if the column lattice drifted off the printed
        # grid (equal division over a bad bright extent), pin ALL cards to
        # the sheet's strong cross-card lattice.  Fires only when the
        # dip/Hough and homography paths both declined (cb is still the raw
        # equal division), the equal-division columns barely align with the
        # printed dividers (<3 of 4), and the cross-card lattice is also
        # confirmed by the card's OWN vertical lines (on sheets whose cards
        # are header-aligned, one confirming line suffices -- the printed
        # grid is identical across the stacked cards).  Cards already on the
        # grid keep their proven columns, so approved sheets stay
        # byte-stable.  The lattice itself is only computed when a super-
        # majority of strong clusters sit on one uniform grid anchored to
        # the sheet's left border (see _best_column_lattice).
        rewired = False
        if lattice is not None:
            aligned = sum(1 for k in range(1, 5)
                          if cb[k] is not None
                          and any(abs(cb[k] - x) <= 12 for x, _ in strong))
            # Only rewire a card whose columns are STILL the raw equal
            # division (the dip/Hough and homography paths both declined it)
            # and that is clearly off the printed grid but the shared lattice
            # is confirmed by the card's OWN vertical lines.
            if (cb == cb_eq and aligned < 3):
                lx0, lp = lattice
                lx = [lx0 + lp * k for k in range(6)]
                if 0 <= lx[0] and lx[-1] < gray.shape[1]:
                    own = sum(1 for k in range(1, 5)
                              if any(abs(lx[k] - h) <= 6 for h in vx_abs))
                    if own >= 2 or (sheet_aligned and own >= 1):
                        cb = list(lx)
                        rewired = True
                        if verbose:
                            print(f"card{i + 1}: columns rewired to sheet "
                                  f"lattice {cb}")

        # ---- ROWS: dip + snap, then enforce uniform lattice on snapped boundaries ----
        # (snap already done above for rb)
        # Rebuild the lattice from the interior dividers only, so the region
        # bounds (esp. the last card's bottom = image edge) can't stretch the
        # rows and a single mis-snapped divider can't skew the pitch.
        if len(rb) == 6:
            rb = _fix_row_lattice(rb, grid_sig, top, bottom, gray, x0, x1)
            gap_ar = np.diff(rb[1:5])
            prev_pitch2 = prev_pitch
            prev_pitch = float(np.median(gap_ar)) if gap_ar.size else None

        if verbose:
            print(f"card{i + 1}: band ({x0},{y0})-({x1},{y1}) "
                  f"card top={top} bottom={bottom} "
                  f"rows={rb} cols={cb}")

        if trace is not None:
            trace.setdefault("cards", []).append({
                "card": i + 1,
                "top": top,
                "bottom": bottom,
                "rows": [int(v) for v in rb],
                "cb_eq": [int(v) for v in cb_eq],
                "cb_used": [int(v) for v in cb],
                "hom": (list(hom_cols[i]) if hom_cols is not None
                        and hom_cols[i] is not None else None),
                "use_hom": bool(use_hom),
                "rewired": bool(rewired)})

        for r in range(5):
            for c in range(5):
                yA, yB, xA, xB = rb[r], rb[r + 1], cb[c], cb[c + 1]
                if yB - yA <= 2 * INSET or xB - xA <= 2 * INSET:
                    continue
                cell = img[yA + INSET:yB - INSET, xA + INSET:xB - INSET]
                if cell.size == 0:
                    continue
                if return_geometry:
                    cells_out.append((i + 1, r, c, cell, (xA, yA, xB, yB)))
                else:
                    cells_out.append((i + 1, r, c, cell))

    if trace is not None:
        trace["x0"] = int(grid_x0)
        trace["x1"] = int(grid_x1)
        trace["boxes"] = [[int(v) for v in b] for b in boxes]
        trace["clusters"] = [[int(v) for v in b] for b in strong]
        trace["lattice"] = ([round(lattice[0], 1), round(lattice[1], 1)]
                            if lattice is not None else None)
        trace["hom_all"] = ([list(h) if h is not None else None for h in hom_cols]
                            if hom_cols is not None else None)
    return cells_out


def sheet_to_cells_teal(img, verbose=False, return_geometry=False,
                        trace=None):
    """Card geometry derived from teal header bboxes (auto-detected sheet
    print color, see teal_card_bboxes).  See _sheet_to_cells_with_boxes."""
    return _sheet_to_cells_with_boxes(
        img, teal_card_bboxes(img), verbose, return_geometry, trace)


def _grid_detect_hough_extent(gray, header_boxes):
    """Per-card Hough outer-border detection, used to recover the PRINTED
    grid extent when brightness-based paper detection is ambiguous (e.g. a
    bright desk right of the sheet makes the bright-paper column run extend
    onto the desktop).

    Runs find_grid_line_positions with edge_exclude_frac=0 so the card's own
    outer border lines are eligible (they're normally dropped as 'edge').
    A card counts as confident when at least 3 vertical lines are found and
    the LEFTMOST detected line sits near the card's header-left (within 15%
    of the header width) — a real grid border, unlike a desk/shadow line.
    Returns (median_left, max_right) across confident cards, or None."""
    from pipeline import find_grid_line_positions

    H, W = gray.shape[:2]
    # Crop to the central 76% of the FRAME width: wide enough to contain
    # the printed grid (even when a header box is clipped), but narrow
    # enough to exclude the desk/shadow lines near the photo's edge that
    # mimick full-height vertical lines.
    cx0, cx1 = int(0.12 * W), int(0.88 * W)

    left_borders, right_borders = [], []
    for i, (bx0, by0, bx1, by1) in enumerate(header_boxes):
        top = by1 + 2
        bottom = header_boxes[i + 1][1] if i + 1 < len(header_boxes) else H
        if bottom - top < 60:
            continue
        crop = cv2.cvtColor(gray[top:bottom, cx0:cx1], cv2.COLOR_GRAY2BGR)
        _, vx = find_grid_line_positions(crop, edge_exclude_frac=0)
        full = [v + cx0 for v in vx]
        if len(full) < 3:
            continue
        lo = full[0]
        tol = 0.15 * (bx1 - bx0)
        if abs(lo - bx0) <= tol:
            # A real grid border, unlike a desk/shadow line: the detected
            # vertical line nearest the card's header edge.
            left_borders.append(lo)
            best = min(full, key=lambda x: abs(x - bx1))
            if abs(best - bx1) <= 0.30 * (bx1 - bx0):
                right_borders.append(best)
    if not left_borders and not right_borders:
        return None
    med_left = int(np.median(left_borders)) if left_borders else None
    max_right = max(right_borders) if right_borders else None
    return med_left, max_right


def _frame_bright_extent(gray, header_boxes=None):
    """Sheet's bright-paper column extent over the frame, used to
    seed the column bounds when no header color is detectable.

    If header_boxes are provided, uses their union to define the search range,
    then analyzes the middle third of the image within that range.  The
    left edge comes from that brightness analysis; the right edge is capped
    by the Hough-detected printed grid border when one is found, so a bright
    desk right of the sheet can't extend the columns off the paper."""
    h = gray.shape[0]
    
    if header_boxes is not None and len(header_boxes) > 0:
        header_x0 = min(b[0] for b in header_boxes)
        header_x1 = max(b[2] for b in header_boxes)
        
        # Analyze the middle third of the image within the union range
        y0, y1 = h // 3, 2 * h // 3
        band = gray[y0:y1, header_x0:header_x1]
        colmean = band.mean(axis=0).astype(np.uint8)
        thr, _ = cv2.threshold(colmean, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = colmean > max(40, int(thr) * 0.90)
        xs = np.where(bright)[0]
        if xs.size > 0:
            x0 = header_x0 + int(xs[0])
            x1 = header_x0 + int(xs[-1])
            grid = _grid_detect_hough_extent(gray, header_boxes)
            if grid is not None:
                grid_left, grid_right = grid
                if grid_right is not None and grid_right < x1:
                    x1 = min(x1, grid_right)
                if grid_left is not None:
                    lefts = [b[0] for b in header_boxes]
                    # The bright left is suspect when it sits at/before the
                    # narrowest header: a bleedy header pulls the bright run
                    # off the printed grid.  In that case pull the left edge
                    # back to the Hough-detected printed border.
                    if x0 <= min(lefts) + 6 and (max(lefts) - min(lefts)) >= 12:
                        x0 = max(x0, grid_left)
            return x0, x1
    
    # Fallback: middle third of full image
    y0, y1 = h // 3, 2 * h // 3
    band = gray[y0:y1, :]
    colmean = band.mean(axis=0).astype(np.uint8)
    thr, _ = cv2.threshold(colmean, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = colmean > max(40, int(thr) * 0.90)
    xs = np.where(bright)[0]
    if xs.size == 0:
        return None
    return int(xs[0]), int(xs[-1])




def _frame_bright_extent_card(gray, hx0, hx1, top, bottom):
    """Per-card bright extent from the card body vertical range.
    Finds the LEFTMOST bright region (card paper) within the header x-range,
    ignoring table/desk to the right.  The card paper is typically the
    leftmost significant bright region in the card body."""
    h = gray.shape[0]
    y0 = top + (bottom - top) // 3
    y1 = top + 2 * (bottom - top) // 3
    if y1 <= y0:
        y0, y1 = top, bottom
    band = gray[y0:y1, hx0:hx1]
    if band.size == 0:
        return None
    colmean = band.mean(axis=0).astype(np.uint8)
    thr, _ = cv2.threshold(colmean, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = colmean > max(40, int(thr) * 0.90)
    xs = np.where(bright)[0]
    if xs.size == 0:
        return None
    # Find LEFTMOST significant bright run (card paper is leftmost)
    # Group contiguous bright pixels into runs
    runs = []
    in_run = False
    start = 0
    for i, b in enumerate(bright):
        if b and not in_run:
            in_run = True
            start = i
        elif not b and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(bright) - 1))
    # Filter to runs with sufficient width (>= 50px)
    runs = [(s, e) for s, e in runs if e - s >= 50]
    if not runs:
        # Fallback: full bright extent
        xs = np.where(bright)[0]
        if xs.size == 0:
            return None
        return int(hx0 + xs[0]), int(hx0 + xs[-1])
    # Use LEFTMOST run that starts near the left edge of the header
    # (card paper is typically at the left edge of the card)
    # Prefer runs that start within 100px of the left edge
    lefts = [s for s, e in runs]
    best_idx = 0
    for i, s in enumerate(lefts):
        if s <= min(lefts) + 100:
            best_idx = i
            break
    s, e = runs[best_idx]
    return int(hx0 + s), int(hx0 + e)



def forced_card_bboxes(img, band_tops):
    """Build 3 card-band boxes from user-tapped rows (each tap = the TOP
    edge of a card's gray header bar, in the normalized image's pixel
    space).  The card body top is band_top + band height, where band
    height is a fixed fraction of the card pitch (the printed header bars
    are a consistent ~0.13 of card pitch across sheet colors); the column
    extent comes from the card's bright-paper body (see _card_x_extent),
    independent of the washed-out header color.

    All three cards are one physical sheet, so they share one column
    extent.  The per-card bright-body read (y-dependent) fragments when
    the sheet is side-shadowed or washed out, producing wildly different
    left edges (e.g. 516/873/933) that later shift every card's columns
    off the printed grid; clamp each value back to the frame-level extent
    whenever it drifts too far from it."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tops = sorted(int(round(min(H - 2, max(2, float(t)))))
                  for t in band_tops)
    if len(tops) < 3:
        return []
    ext = _frame_bright_extent(gray)
    if ext is None:
        return []
    sx0, sx1 = ext
    tol = 50
    cx = (sx0 + sx1) / 2
    boxes = []
    for i, t in enumerate(tops):
        pitch = (tops[1] - tops[0]) if i == 0 else (tops[i] - tops[i - 1])
        if i + 1 < len(tops):
            next_t = tops[i + 1]
        else:
            next_t = min(H, t + int(round(1.25 * max(120, pitch))))
        h = int(max(30, min(90, round(0.13 * pitch))))
        y1 = min(H - 2, t + h)
        xL, xR = _card_x_extent(gray, y1 + 2, next_t, cx)
        if xL < 0 or abs(xL - sx0) > tol:
            xL = sx0
        if xR < 0 or abs(xR - sx1) > tol:
            xR = sx1
        boxes.append((int(xL), t, int(xR), int(y1)))
    return boxes


def sheet_to_cells_forced(img, band_tops, verbose=False,
                          return_geometry=False, trace=None):
    """Card geometry pinned to user-tapped band tops -- the assisted-scan
    path for washed-out photos (e.g. gray sheets) where no header color
    gate can find the card bands on its own."""
    return _sheet_to_cells_with_boxes(
        img, forced_card_bboxes(img, band_tops), verbose, return_geometry,
        trace)


def _card_x_extent(gray, y0, y1, cx):
    """Bright-paper horizontal extent of the card region: among contiguous
    runs of bright columns, pick the one containing cx (the teal band's
    center), preferring it over any background bright region."""
    band = gray[y0:min(y1, gray.shape[0]), :]
    if band.size == 0:
        return -1, -1
    colmean = band.mean(axis=0)
    c8 = colmean.astype(np.uint8)
    thr, _ = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = c8 > max(40, int(thr) * 0.90)
    runs = []
    cur = None
    for x, b in enumerate(bright):
        if b and cur is None:
            cur = [x, x]
        elif b:
            cur[1] = x
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    runs = [r for r in runs if r[1] - r[0] >= 30]
    if not runs:
        return -1, -1
    candidate = runs[0]
    for r in runs[1:]:
        if (abs((r[0] + r[1]) / 2 - cx) < abs((candidate[0] + candidate[1]) / 2 - cx)
                or (r[1] - r[0]) > (candidate[1] - candidate[0]) * 2):
            candidate = r
    return candidate[0], candidate[1]


if __name__ == "__main__":
    out = Path("/tmp/photo_cells")
    out.mkdir(exist_ok=True)
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        cells = sheet_to_cells_photo(img, verbose=True)
        print(f"{p}: {len(cells)} cells")
        for cid, r, c, cell in cells:
            cv2.imwrite(str(out / f"{Path(p).stem}_c{cid}_r{r}c{c}.jpg"), cell)