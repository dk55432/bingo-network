#!/usr/bin/env python3
"""
Extract 5×5 bingo cells from a photographed sheet that may contain
one or more cards stacked vertically on a single paper strip.

Strategy
--------
1. Detect the outer paper strip by locating the dense region of
   horizontal + vertical grid lines (unique to the bingo cards).
2. Perspective-warp the whole strip (nearly axis-aligned in practice).
3. Split the warped strip into individual cards via horizontal gutters
   or equal-height fallback.
4. Extract the regular 5×5 grid from each card.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def order_points(points):
    """Return points in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    result[0] = points[np.argmin(sums)]   # top-left
    result[2] = points[np.argmax(sums)]   # bottom-right
    result[1] = points[np.argmin(diffs)]  # top-right
    result[3] = points[np.argmax(diffs)]  # bottom-left
    return result


def four_point_warp(image, corners, output_size=None):
    """
    Perspective-correct a quadrilateral.
    When output_size is None the destination size is derived from the
    average side lengths of the ordered corners.
    """
    corners = order_points(corners)

    if output_size is None:
        (tl, tr, br, bl) = corners
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        width = max(int(width_a), int(width_b), 1)
        height = max(int(height_a), int(height_b), 1)
        # Cap working resolution while preserving aspect
        scale = min(1400 / max(width, 1), 2000 / max(height, 1), 1.0)
        width = max(int(width * scale), 200)
        height = max(int(height * scale), 200)
    else:
        width, height = output_size

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


# ---------------------------------------------------------------------------
# Deskewing
# ---------------------------------------------------------------------------

def estimate_skew_angle(card, max_angle=10.0):
    """
    Estimate the residual rotation of a card image (degrees) from the
    median angle of its long near-horizontal line segments.

    The strip warp uses axis-aligned corners, so any tilt/perspective in
    the photograph survives into the cropped cards. Even 1-2 degrees
    smears a row divider across tens of projection rows, which is what
    makes the horizontal-line detection miss dividers (and eventually
    slice straight through digits).
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=80,
        minLineLength=int(w * 0.35), maxLineGap=25,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if abs(ang) < max_angle:
            angles.append(ang)
    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def deskew(card, min_angle=0.3):
    """Rotate a card image to cancel its estimated residual tilt."""
    angle = estimate_skew_angle(card)
    if abs(angle) < min_angle:
        return card
    h, w = card.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        card, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Sheet (outer paper strip) detection – grid-line based
# ---------------------------------------------------------------------------

def detect_sheet(image, debug_path=None):
    """
    Locate the outer boundary of the bingo paper strip by finding the
    dense vertical band of horizontal/vertical grid lines.

    Returns (ordered_corners, debug_mask) or (None, mask) on failure.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Local contrast normalisation helps with uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Black lines on light paper → binary-inverted adaptive threshold.
    binary = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 8
    )

    # Extract long horizontal and vertical line segments.
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1))
    vert_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 20)))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_k)
    vert  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_k)

    # --- X bounds from the column projection of horizontal lines ----------
    col_score = np.sum(horiz > 0, axis=0).astype(np.float32)
    col_score = cv2.GaussianBlur(col_score.reshape(1, -1), (1, 31), 0).ravel()

    peak = int(col_score.argmax())
    if col_score[peak] < 10:
        return None, horiz   # no meaningful grid detected

    thresh_val = col_score[peak] * 0.25
    high = col_score > thresh_val
    # Take the contiguous run that contains the peak.
    left = peak
    while left > 0 and high[left]:
        left -= 1
    left += 1
    right = peak
    while right < w - 1 and high[right]:
        right += 1

    # Guard against a degenerate band.
    if right - left < w * 0.12:
        return None, horiz

    # --- Y bounds from any line activity inside a slightly wider band -----
    band = slice(max(0, left - 30), min(w, right + 30))
    row_has = (
        (np.sum(horiz[:, band] > 0, axis=1) > 1) |
        (np.sum(vert[:, band]  > 0, axis=1) > 1)
    ).astype(np.uint8)

    # Dilate the 1-D signal (wide 1-D kernel) so neighbouring grid lines
    # and the three cards merge into continuous runs.
    dilate_width = 181  # ~90 px each side
    kernel_1d = np.ones((1, dilate_width), np.uint8)
    row_dil = cv2.dilate(row_has.reshape(1, -1), kernel_1d).ravel()

    # Find the longest contiguous True run.
    runs = []
    start = None
    for i, v in enumerate(row_dil > 0):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(row_dil) - 1, len(row_dil) - start))

    if not runs:
        return None, horiz

    runs.sort(key=lambda r: -r[2])
    top, bottom = runs[0][0], runs[0][1]

    if bottom - top < h * 0.35:
        return None, horiz

    # Axis-aligned corners (the photograph is nearly straight-on).
    # Clipping guarantees we never produce coordinates outside the image.
    left   = int(np.clip(left,   0, w - 1))
    right  = int(np.clip(right,  0, w - 1))
    top    = int(np.clip(top,    0, h - 1))
    bottom = int(np.clip(bottom, 0, h - 1))

    corners = np.array([
        [left,  top],
        [right, top],
        [right, bottom],
        [left,  bottom],
    ], dtype=np.float32)

    ordered = order_points(corners)

    if debug_path:
        debug = image.copy()
        cv2.polylines(debug, [ordered.astype(np.int32)], True, (0, 255, 0), 6)
        labels = ["TL", "TR", "BR", "BL"]
        for pt, lab in zip(ordered, labels):
            p = tuple(pt.astype(int))
            cv2.circle(debug, p, 14, (0, 0, 255), -1)
            cv2.putText(debug, lab, (p[0] + 12, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.imwrite(str(debug_path), debug)

    # Return a useful debug mask (the horizontal lines).
    return ordered, horiz


# ---------------------------------------------------------------------------
# Split warped strip into individual cards
# ---------------------------------------------------------------------------

def split_sheet_into_cards(warped_sheet, number_of_cards=3, gap_ratio=0.015):
    """Equal-height split with a small inter-card gap."""
    h, w = warped_sheet.shape[:2]
    gap = int(h * gap_ratio)
    usable = h - gap * (number_of_cards - 1)
    card_h = usable // number_of_cards

    cards = []
    for i in range(number_of_cards):
        y1 = i * (card_h + gap)
        y2 = y1 + card_h
        y1 = max(0, y1)
        y2 = min(h, y2)
        cards.append(warped_sheet[y1:y2].copy())
    return cards


def split_using_horizontal_gaps(warped_sheet, number_of_cards=3):
    """
    Locate low-variation horizontal bands (the gutters between cards)
    and split on the strongest internal gaps.
    Returns a list of card images or None if detection fails.
    """
    gray = cv2.cvtColor(warped_sheet, cv2.COLOR_BGR2GRAY)
    row_std = np.std(gray.astype(np.float32), axis=1)
    smooth = cv2.GaussianBlur(row_std.reshape(-1, 1), (1, 31), 0).ravel()

    threshold = np.percentile(smooth, 18)
    is_gap = smooth < threshold

    runs = []
    start = None
    for y, g in enumerate(is_gap):
        if g and start is None:
            start = y
        elif not g and start is not None:
            if y - start >= 4:
                runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_gap) - 1))

    if len(runs) < number_of_cards - 1:
        return None

    centers = sorted((a + b) // 2 for a, b in runs)
    h = len(gray)

    from itertools import combinations
    best = None
    for chosen in combinations(centers, number_of_cards - 1):
        cuts = [0] + list(chosen) + [h]
        sizes = [cuts[i + 1] - cuts[i] for i in range(number_of_cards)]
        if min(sizes) < h * 0.12:
            continue
        score = min(sizes)
        if best is None or score > best[0]:
            best = (score, cuts)

    if best is None:
        return None

    _, cuts = best
    return [warped_sheet[cuts[i]:cuts[i + 1]].copy()
            for i in range(number_of_cards)]


def split_using_pink_headers(warped_sheet, number_of_cards=3):
    """
    Split the warped strip by locating the pink BINGO header bands.
    Each card starts at (or just before) its header and ends just
    before the next header (or at the end of content).
    """
    h, w = warped_sheet.shape[:2]
    hsv = cv2.cvtColor(warped_sheet, cv2.COLOR_BGR2HSV)

    # Distinctive pink of the BINGO headers
    pink = (
        ((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 162)) &
        (hsv[:, :, 1] > 45) &
        (hsv[:, :, 2] > 140)
    )
    row_score = np.sum(pink, axis=1).astype(np.float32)
    row_s = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 21), 0).ravel()

    # Strong, well-separated peaks (only the true saturated headers)
    peaks = []
    for i in np.argsort(row_s)[::-1]:
        # Require a high absolute score so weak residual bands are ignored
        if row_s[i] < max(120, row_s.max() * 0.35):
            break
        if all(abs(i - p) > 200 for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= number_of_cards + 1:
            break
    peaks = sorted(peaks)

    if len(peaks) < number_of_cards:
        return None

    headers = peaks[:number_of_cards]

    # End of printable content (avoid dark table at the bottom)
    gray = cv2.cvtColor(warped_sheet, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)
    content_end = h - 1
    for y in range(h - 1, h // 2, -1):
        if row_mean[y] > 70:
            content_end = min(h - 1, y + 10)
            break

    # Build card ranges: start a few pixels before each header
    starts = []
    ends = []
    for i, hy in enumerate(headers):
        start = max(0, hy - 8) if i > 0 else max(0, hy - 15)
        if i + 1 < len(headers):
            end = headers[i + 1] - 5
        else:
            end = content_end
        starts.append(start)
        ends.append(end)

    # Sanity: each card should be a reasonable fraction of the strip
    heights = [e - s for s, e in zip(starts, ends)]
    if min(heights) < h * 0.12:
        return None

    cards = [warped_sheet[s:e].copy() for s, e in zip(starts, ends)]
    return cards


# ---------------------------------------------------------------------------
# Grid / cell extraction
# ---------------------------------------------------------------------------

def _crop_to_number_grid(card):
    """
    Crop to the 5×5 number area.

    Y bounds come from the cluster of strong horizontal grid lines.
    X bounds use a modest fixed side crop (the strip warp already
    centres the paper, so a percentage trim is reliable).
    Returns (roi, y_offset, x_offset).
    """
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    # --- Horizontal lines → top / bottom of the number grid ----------------
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 5), 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    prof = cv2.GaussianBlur(
        horiz.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 5), 0
    ).ravel()

    min_dist = max(12, h // 14)
    peaks = []
    for i in np.argsort(prof)[::-1]:
        if prof[i] < prof.max() * 0.22:
            break
        if all(abs(i - p) >= min_dist for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= 8:
            break
    peaks = sorted(peaks)

    if len(peaks) >= 4:
        med = float(np.median(np.diff(peaks)))
        top = max(0, int(peaks[0] - med * 0.55))
        bottom = min(h, int(peaks[-1] + med * 0.55))
    else:
        top = int(h * 0.14)
        bottom = int(h * 0.94)

    top = int(np.clip(top, 0, h // 3))
    bottom = int(np.clip(bottom, top + h // 4, h))

    # --- Side crop (fixed fraction – reliable after the strip warp) --------
    left = max(4, int(w * 0.04))
    right = w - left

    roi = card[top:bottom, left:right]
    return roi, top, left



def _fit_regular_rows(candidates, h, rows, min_span_frac=0.45, max_span_frac=1.02,
                      miss_penalty=0.35, prior_pitch=None):
    """
    Choose `rows + 1` row boundaries from candidate divider positions by
    fitting an evenly-spaced model, tolerating missing dividers.

    On real cards the outer borders often print (much) fainter than the
    internal dividers and may vanish from the candidate list entirely,
    so models are NOT restricted to candidate pairs spanning the whole
    grid: every candidate pair is tried at every pair of row indices
    (a, b) — adjacent dividers included — extrapolating the remaining
    boundaries outward from the step implied by that assignment. Each
    expected boundary then snaps
    to the nearest candidate within a fraction of a row height, or
    stays at its predicted position when none is found (a faint or
    broken divider). Unmatched positions cost a fixed penalty, so a
    model explaining many real lines beats a short-span model that
    merely sits on the few strongest ones.

    When only sparse dividers survive, several index assignments can
    explain them equally well while implying different row heights
    (aliasing). The digit-ink autocorrelation pitch breaks those ties:
    if any model agrees with it, models at wildly different steps are
    discarded.

    Returns `rows + 1` y positions, or None if nothing plausible fits.
    """
    n = rows + 1
    if not candidates:
        return None

    def single_anchor_models(c0):
        # A single anchor can still pin the model when the digit-ink
        # autocorrelation supplies the pitch (faint vertical dividers
        # routinely leave exactly one detectable line).
        out = []
        if prior_pitch:
            for a in range(n):
                top = c0 - a * prior_pitch
                bottom = top + (n - 1) * prior_pitch
                if min_span_frac * h <= bottom - top <= max_span_frac * h \
                        and -0.02 * h <= top and bottom <= h * 1.02:
                    out.append((prior_pitch, c0, c0, a, a))
        return out

    models = []
    if len(candidates) == 1:
        models.extend(single_anchor_models(candidates[0][0]))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ci, cj = candidates[i][0], candidates[j][0]
            for a in range(n - 1):
                for b in range(a + 1, n):
                    step = (cj - ci) / (b - a)
                    if step < 5:
                        continue
                    top = ci - a * step
                    bottom = cj + (n - 1 - b) * step
                    span = bottom - top
                    if not (min_span_frac * h <= span <= max_span_frac * h):
                        continue
                    if top < -0.02 * h or bottom > h * 1.02:
                        continue
                    models.append((step, ci, cj, a, b))

    # Pair anchors clustered in one corner of the axis cannot produce a
    # plausible grid span at all — every pair model is rejected and the
    # fit would fail even though a pitched model through any single
    # anchor is perfectly viable. Fall back to single-anchor hypotheses.
    if not models and prior_pitch:
        for c in candidates:
            models.extend(single_anchor_models(c[0]))

    # The pitch prior disambiguates aliasing (several index assignments
    # explain the same sparse lines at different steps). A tight window
    # is what actually kills aliased models — e.g. a 208px step that is
    # really a 246px grid read one divider off — but the autocorrelation
    # peak itself can be a few percent off, so relax rather than fail.
    if prior_pitch and models:
        for tol in (0.10, 0.16, 1.0):
            near = [m for m in models
                    if abs(m[0] - prior_pitch) <= tol * prior_pitch]
            if near:
                models = near
                break

    # Snapping tolerance as a fraction of the pitch, scaled by candidate
    # strength. Real boundaries sit within a few px of their printed
    # line / gutter minimum, while an aliased model can be tens of px
    # off; a generous window lets wrong models harvest distant
    # candidates as fake support. Weak evidence (deep-but-unplaced ink
    # valleys can be whitespace inside a cell rather than a gutter) gets
    # half the window so it can fine-tune a nearby boundary but never
    # pull one across half a column.
    snap_frac = 0.12

    def snap_window(step_, cov):
        return step_ * snap_frac * max(0.5, min(1.0, cov / 0.25))

    best = None
    for step, ci, cj, a, b in models:
        top = ci - a * step
        bottom = cj + (n - 1 - b) * step
        score, positions = 0.0, []
        for k in range(n):
            expected = top + k * step
            near = [(c[0], c[1]) for c in candidates
                    if abs(c[0] - expected) <= snap_window(step, c[1])]
            if near:
                pos, cov = max(near, key=lambda c: c[1])
                score += cov
            else:
                pos = expected
                score -= miss_penalty
            positions.append(pos)

        if not all(positions[k] < positions[k + 1] for k in range(n - 1)):
            continue
        score /= n
        if best is None or score > best[0]:
            best = (score, positions, step)

    if best is None:
        return None

    # Least-squares refit of the winning model over every MATCHED line
    # (not just its anchor pair): several (pair, index-assignment)
    # combinations explain the same lines equally well but extrapolate
    # different outer boundaries; fitting all the evidence settles them
    # deterministically and tracks the slight row-height drift caused by
    # residual perspective. Matches are weighted by coverage so faint
    # dips cannot bend a model anchored on solid printed lines.
    _, positions, step = best
    matched = []
    for k, p in enumerate(positions):
        hits = [(c[0], c[1]) for c in candidates
                if abs(p - c[0]) <= snap_window(step, c[1])]
        if hits:
            pos, cov = max(hits, key=lambda c: c[1])
            matched.append((k, pos, cov))
    if len(matched) >= 2:
        ks = np.array([k for k, _, _ in matched], dtype=np.float64)
        ys = np.array([p for _, p, _ in matched], dtype=np.float64)
        ws = np.array([c for _, _, c in matched], dtype=np.float64)
        slope, intercept = np.polyfit(ks, ys, 1, w=ws)
        refit = []
        for k in range(n):
            expected = intercept + k * slope
            near = [(c[0], c[1]) for c in candidates
                    if abs(c[0] - expected) <= snap_window(abs(slope), c[1])]
            refit.append(max(near, key=lambda c: c[1])[0] if near else expected)
        if all(refit[k] < refit[k + 1] for k in range(n - 1)):
            positions = refit
    else:
        slope = abs(step)

    # An extrapolated outer boundary is uncertain by nature — nudge it
    # slightly outward so uncertainty costs white margin (harmless for
    # the digit classifier) instead of a sliced digit tail.
    pad = 0.06 * abs(slope)
    if not any(abs(positions[0] - c[0]) <= snap_window(abs(slope), c[1])
               for c in candidates):
        positions[0] -= pad
    if not any(abs(positions[-1] - c[0]) <= snap_window(abs(slope), c[1])
               for c in candidates):
        positions[-1] += pad
    return [int(round(p)) for p in positions]


def _fit_vertical_dividers(gray, y0, y1, w, columns):
    """
    Locate the `columns + 1` vertical dividers of the number grid that
    occupies rows y0:y1, mirroring the horizontal-divider strategy:
    printed-line evidence + an evenly-spaced fit with digit-ink support.

    Vertical grid lines print much fainter than horizontal ones and
    routinely drop out entirely, so candidates come from two sources:

    * morphological opening at two kernel scales (a long kernel misses
      partially-printed lines a short one still catches);
    * deep valleys in the per-column digit-ink profile — gutters between
      columns stay ink-free except where a divider prints, so their
      minima bracket the true boundaries. Fed in as weak candidates they
      let the fitter snap extrapolated boundaries onto real content and
      give the least-squares refit more matches to work with.

    The column pitch prior comes from autocorrelating the same ink
    profile — the transpose of what disambiguates the row fit.

    Returns `columns + 1` x positions, or None if nothing plausible fits.
    """
    band = gray[y0:y1, :]
    bh = band.shape[0]
    vb = cv2.adaptiveThreshold(
        band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    # Column ink profile: shared by line coverage, dips and pitch prior.
    ink = vb.mean(axis=0).astype(np.float32) / 255.0
    sm = np.convolve(ink, np.ones(9) / 9, mode="same")

    ac = np.correlate(sm - sm.mean(), sm - sm.mean(), mode="full")
    ac = ac[len(sm) - 1:]
    lo, hi = int(w * 0.08), max(int(w * 0.08) + 1, int(w * 0.30))
    pitch = None
    if hi > lo and ac[0] > 0:
        window = ac[lo:hi]
        if float(window.max()) > 0.05 * float(ac[0]):
            pitch = float(np.argmax(window) + lo)

    # --- Printed-line candidates ------------------------------------------
    raw = []
    for frac in (0.25, 0.10):
        kh = max(20, int(frac * bh))
        vert = cv2.morphologyEx(
            vb, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)),
        )
        vprof = vert.sum(axis=0).astype(np.float32) / 255.0 / bh
        floor = max(0.12, 0.25 * float(vprof.max()))
        if vprof.max() <= floor or vprof.max() < 1e-6:
            continue
        max_run = max(8, int(0.02 * w))
        x = 0
        while x < w:
            if vprof[x] > floor:
                run_start = x
                while x + 1 < w and vprof[x + 1] > floor:
                    x += 1
                touches_edge = run_start <= 7 or x >= w - 8
                too_thick = (x - run_start + 1) > max_run
                if not touches_edge and not too_thick:
                    peak_x = run_start + int(np.argmax(vprof[run_start:x + 1]))
                    raw.append((peak_x, float(vprof[peak_x])))
            x += 1

    # Merge hits from both scales (positions differ by a few px).
    raw.sort(key=lambda c: -c[1])
    lines = []
    for px, cov in raw:
        if all(abs(px - lx) > max(15, int(0.06 * w)) for lx, _ in lines):
            lines.append((px, cov))
    lines.sort()

    # --- Gutter-dip candidates (weak) --------------------------------------
    # Gutters show up as valleys of the ink profile, but how deep they
    # sit depends on the card's print darkness, and deep whitespace
    # pockets INSIDE a cell can rival a faint gutter in absolute depth.
    # What distinguishes a real inter-column gutter is prominence: the
    # profile rises towards digit bands on BOTH sides. Rank local minima
    # by prominence, greedily keep the best-separated ones. The grid has
    # only columns + 1 boundaries; spurious extras lose scoring ties
    # against printed lines and only fine-tune nearby positions.
    dips = []
    if pitch:
        min_sep = max(20, int(0.35 * pitch))
        reach = int(0.60 * pitch)
        scored = []
        for x in range(8, w - 8):
            if sm[x] <= sm[x - 1] and sm[x] <= sm[x + 1]:
                lpeak = float(sm[max(0, x - reach):x].max())
                rpeak = float(sm[x + 1:min(w, x + reach + 1)].max())
                scored.append((min(lpeak, rpeak) - float(sm[x]), x))
        scored.sort(reverse=True)
        for prom, x in scored:
            if len(dips) >= columns + 2:
                break
            if all(abs(x - dx) > min_sep for dx, _ in dips):
                dips.append((x, prom))
        dips.sort()
    dip_strength = 0.12

    candidates = [(int(px), cov) for px, cov in lines]
    candidates += [(int(dx), dip_strength) for dx, _ in dips]
    candidates.sort()

    positions = _fit_regular_rows(candidates, w, columns, prior_pitch=pitch)
    if positions is None:
        return None

    # Guard against dividers drifting outside the card.
    return [int(np.clip(x, 0, w - 1)) for x in positions]


def extract_grid_cells(warped, rows=5, columns=5):
    """
    Extract the 5×5 cells from a single card image.

    Horizontal lines: morphological projection + regular-spacing fit
    (tolerates faint or broken dividers).
    Vertical lines: the same strategy transposed — printed-line
    candidates plus gutter dips fitted with a regular-spacing model
    whose pitch prior comes from digit-ink autocorrelation; falls back
    to digit-centroid clustering only when that fails.

    Returns (cells, x_lines, y_lines) in the card's coordinate system.
    """
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    # ----- Horizontal grid lines ------------------------------------------
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 4), 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    # Reconnect dividers broken by residual tilt or faint printing before
    # projecting — a line smeared across 3-5 rows otherwise vanishes from
    # the per-row projection entirely.
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    horiz = cv2.dilate(horiz, vert_k)
    hprof = horiz.sum(axis=1).astype(np.float32) / 255.0  # width coverage 0..w
    hprof = cv2.GaussianBlur(hprof.reshape(-1, 1), (1, 5), 0).ravel()

    # Candidate dividers: contiguous runs above an absolute floor (a real
    # divider spans most of the card's width), reduced to their peak.
    # Coverage is normalised to 0..1 of the card width. Runs hugging the
    # image edge (deskew border-replication slivers, the dark table
    # beyond the paper) or thicker than a printed line (a filled region,
    # not a divider) are rejected — both otherwise masquerade as strong
    # grid lines and drag the row fit into slicing through digits.
    floor = max(0.12, 0.25 * float(hprof.max()) / w)
    max_run = max(8, int(0.02 * h))
    candidates = []
    y = 0
    while y < h:
        if hprof[y] > floor * w:
            run_start = y
            while y + 1 < h and hprof[y + 1] > floor * w:
                y += 1
            touches_edge = run_start <= 7 or y >= h - 8
            too_thick = (y - run_start + 1) > max_run
            if not touches_edge and not too_thick:
                run = hprof[run_start:y + 1]
                peak_y = run_start + int(np.argmax(run))
                candidates.append((peak_y, float(run.max()) / w))
        y += 1

    # Digit-ink banding gives an independent read of the row pitch:
    # digit bands repeat with the row period even where every printed
    # divider is too faint to detect. Fed to the fitter as a prior, it
    # rules out aliased step sizes when only sparse dividers survive.
    ink = binary.mean(axis=1).astype(np.float32) / 255.0
    ink = np.convolve(ink, np.ones(15) / 15, mode="same")
    ac = np.correlate(ink - ink.mean(), ink - ink.mean(), mode="full")
    ac = ac[len(ink) - 1:]
    lo, hi = int(h * 0.10), max(int(h * 0.10) + 1, int(h * 0.26))
    prior_pitch = None
    if hi > lo and ac[0] > 0:
        window = ac[lo:hi]
        if float(window.max()) > 0.05 * float(ac[0]):
            prior_pitch = float(np.argmax(window) + lo)

    y_lines = _fit_regular_rows(candidates, h, rows, prior_pitch=prior_pitch)

    if y_lines is None:
        y_lines = list(np.linspace(int(h * 0.12), int(h * 0.92), rows + 1).astype(int))
    y_lines = [int(np.clip(y, 0, h - 1)) for y in y_lines]

    # ----- Vertical dividers -------------------------------------------------
    # Primary: printed-line + gutter-dip evidence fitted with a
    # regular-spacing model (_fit_vertical_dividers). Digit-centroid
    # clustering below is only a fallback: its component filters reject
    # many real digits, which starves the k-means of clusters and lets it
    # invent dividers that slice straight through numbers.
    y0 = max(0, y_lines[0] - 5)
    y1 = min(h, y_lines[-1] + 5)
    x_lines = _fit_vertical_dividers(g, y0, y1, w, columns)

    if x_lines is None:
        roi = g[y0:y1, :]
        _, dark = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark, connectivity=8
        )

        digit_cx = []
        for i in range(1, n_labels):
            x, y, bw, bh, area = stats[i]
            cx, cy = centroids[i]
            if not (40 < area < 5000 and 6 < bw < 80 and 12 < bh < 100):
                continue
            if cx < 20 or cx > roi.shape[1] - 20:
                continue
            if cy < 5 or cy > roi.shape[0] - 10:
                continue
            digit_cx.append(float(cx))

        if len(digit_cx) >= columns * 2:
            digit_cx = np.array(digit_cx, dtype=np.float64)
            # 1-D k-means for column centers
            centers = np.linspace(digit_cx.min(), digit_cx.max(), columns)
            for _ in range(25):
                dists = np.abs(digit_cx[:, None] - centers[None, :])
                lab = dists.argmin(axis=1)
                new_c = np.array([
                    digit_cx[lab == k].mean() if np.any(lab == k) else centers[k]
                    for k in range(columns)
                ])
                if np.allclose(centers, new_c, atol=0.5):
                    break
                centers = new_c
            centers = np.sort(centers)

            gaps = np.diff(centers)
            med_gap = float(np.median(gaps)) if len(gaps) else (w * 0.8 / columns)
            dividers = [0.5 * (centers[i] + centers[i + 1]) for i in range(columns - 1)]
            left = centers[0] - med_gap / 2.0
            right = centers[-1] + med_gap / 2.0
            x_lines = [int(round(left))] + [int(round(d)) for d in dividers] + [int(round(right))]
            x_lines = [int(np.clip(x, 0, w - 1)) for x in x_lines]

            # Ensure strictly increasing
            for i in range(1, len(x_lines)):
                if x_lines[i] <= x_lines[i - 1]:
                    x_lines[i] = x_lines[i - 1] + max(10, int(med_gap * 0.4))
            x_lines[-1] = min(x_lines[-1], w - 1)

            # Regularity check — if bad, fall back
            dx = np.diff(x_lines)
            if len(dx) >= 2 and (np.std(dx) / max(np.mean(dx), 1)) > 0.25:
                x_lines = None

    if x_lines is None:
        # Fallback: uniform between outer content bounds
        lefts, rights = [], []
        for y in y_lines:
            yy = int(np.clip(y, 2, h - 3))
            strip = g[yy - 1:yy + 2].min(axis=0)
            dark_idx = np.where(strip < 105)[0]
            if len(dark_idx) > 8:
                lefts.append(int(np.percentile(dark_idx, 3)))
                rights.append(int(np.percentile(dark_idx, 97)))
        left = int(np.median(lefts)) if lefts else int(w * 0.05)
        right = int(np.median(rights)) if rights else int(w * 0.95)
        if right - left < w * 0.4:
            left, right = int(w * 0.05), int(w * 0.95)
        x_lines = list(np.linspace(left, right, columns + 1).astype(int))

    # ----- Collision-escape refinement ---------------------------------------
    # Deskew cancels rotation but not perspective (keystone): printed
    # verticals drift horizontally across the card height, so a global
    # x per divider that clears digits in one row can slice them in
    # another. Rather than hunting for the exact gutter — ambiguous,
    # because whitespace pockets inside a cell rival gutters in ink —
    # each cut only has to STOP COLLIDING: while it touches digit ink
    # it slides right to the first clean corridor, i.e. just past
    # whatever it was slicing. Cuts already sitting in whitespace stay
    # exactly where the fitter put them, preserving grid alignment.
    step = float(np.median(np.diff(x_lines)))
    min_sep = max(8, int(0.05 * step))
    reach = int(0.45 * step)  # max slide; keeps a cut inside its column pair
    clean_T = 0.03            # band-ink fraction under which a column is clear
    clean_run = 4             # consecutive clear columns required

    def first_clean(band, lo, hi):
        """Midpoint of the first >=clean_run clear stretch in [lo, hi),
        with the stretch capped so wide gutters don't fling the cut
        into the middle of the neighbouring cell's whitespace."""
        clear = band <= clean_T
        run = 0
        for x in range(max(0, lo), min(len(band), hi)):
            run = run + 1 if clear[x] else 0
            if run >= clean_run:
                start = x - clean_run + 1
                end = x
                while end + 1 < len(band) and clear[end + 1] \
                        and end - start < max(clean_run, int(0.15 * step)):
                    end += 1
                return (start + end) // 2
        return None

    row_cuts = []
    for r in range(rows):
        y_top = int(np.clip(y_lines[r] + (y_lines[r + 1] - y_lines[r]) // 8, 0, h))
        y_bot = int(np.clip(y_lines[r + 1] - (y_lines[r + 1] - y_lines[r]) // 8, 0, h))
        if y_bot - y_top < 12:
            row_cuts.append(list(x_lines))
            continue
        # Raw grayscale, not the CLAHE copy: CLAHE amplifies paper
        # texture so whitespace stops reading as clean, while digits
        # are dark enough in the plain image for the collision test.
        band = (gray[y_top:y_bot] < 150).mean(axis=0)
        cuts = []
        for k, xp in enumerate(x_lines):
            xp = int(np.clip(xp, 0, w - 1))
            lo = cuts[k - 1] + min_sep if k else 2
            hi = int(x_lines[k + 1]) - min_sep if k + 1 < len(x_lines) else w - 3
            pos = None
            if lo <= xp <= hi and band[xp] > clean_T:
                pos = first_clean(band, max(xp, lo), min(hi + 1, xp + reach))
                if pos is not None and not (lo <= pos <= hi):
                    pos = None
                if pos is None:
                    # No fully clear corridor within reach (dense digits,
                    # or keystone drift larger than the reach): settle for
                    # the least-ink spot so the cut grazes as few digits
                    # as possible instead of standing mid-ink.
                    a = max(lo, int(xp - reach))
                    b = min(hi + 1, int(xp + reach) + 1)
                    if b - a > 10:
                        sm = np.convolve(band[a:b], np.ones(5) / 5, "same")
                        pos = a + int(np.argmin(sm))
            if pos is None:
                # Either already clean (keep the fitted position — it
                # tracks the printed grid) or no clean corridor within
                # reach (keep the fitter's guess rather than wandering
                # into the neighbouring column).
                pos = int(np.clip(xp, lo, hi))
            cuts.append(pos)
        for k in range(1, len(cuts)):
            cuts[k] = max(cuts[k], cuts[k - 1] + min_sep)
        cuts[-1] = min(cuts[-1], w - 1)
        row_cuts.append(cuts)

    # ----- Extract cells with modest inset to avoid grid-line ink ----------
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(columns):
            x1, x2 = row_cuts[r][c], row_cuts[r][c + 1]
            y1, y2 = y_lines[r], y_lines[r + 1]
            mx = max(3, (x2 - x1) // 10)
            my = max(3, (y2 - y1) // 10)
            cell = warped[y1 + my:y2 - my, x1 + mx:x2 - mx]
            row_cells.append(cell)
        cells.append(row_cells)

    return cells, x_lines, y_lines, row_cuts




def grid_quality(card, x_lines, y_lines, row_cuts):
    """
    Cheap sanity read-out for one card's fitted grid: regularity of both
    line families plus median digit-ink sitting on the internal vertical
    cuts. Well-fitted grids on real cards score ink well under 0.05;
    crops that are actually two stacked cards (or worse) alias their
    rows and slice digits everywhere, scoring several times higher.
    """
    h, w = card.shape[:2]
    dy = np.diff(y_lines)
    dx = np.diff(x_lines)
    y_reg = float(np.std(dy) / max(float(np.mean(dy)), 1.0))
    x_reg = float(np.std(dx) / max(float(np.mean(dx)), 1.0))
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    inks = []
    for r in range(len(row_cuts)):
        span = y_lines[r + 1] - y_lines[r]
        y_top = int(np.clip(y_lines[r] + span // 8, 0, h))
        y_bot = int(np.clip(y_lines[r + 1] - span // 8, 0, h))
        if y_bot - y_top < 12:
            continue
        band = (gray[y_top:y_bot] < 150).mean(axis=0)
        for xp in row_cuts[r][1:-1]:
            lo, hi = max(0, int(xp) - 3), min(w, int(xp) + 4)
            inks.append(float(band[lo:hi].mean()))
    return {
        "y_reg": y_reg,
        "x_reg": x_reg,
        "ink": float(np.median(inks)) if inks else 1.0,
    }


def save_cells(cells, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ri, row in enumerate(cells):
        for ci, cell in enumerate(row):
            path = output_dir / f"{prefix}_r{ri + 1}_c{ci + 1}.png"
            cv2.imwrite(str(path), cell)


def save_corner_debug(image, corners, filename):
    debug = image.copy()
    corners = order_points(corners).astype(np.int32)
    cv2.polylines(debug, [corners], True, (0, 0, 255), 6)
    for pt, lab in zip(corners, ["TL", "TR", "BR", "BL"]):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(debug, (x, y), 12, (255, 0, 0), -1)
        cv2.putText(debug, lab, (x + 14, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
    cv2.imwrite(str(filename), debug)


def draw_grid_debug(card, x_lines, y_lines, filename, row_cuts=None):
    debug = card.copy()
    if row_cuts is None:
        for x in x_lines:
            cv2.line(debug, (int(x), 0), (int(x), debug.shape[0] - 1),
                     (0, 0, 255), 2)
    else:
        # True cuts: each divider is drawn only across the row band it
        # was refined for, so keystone drift is visible instead of
        # hidden behind full-height straight lines.
        for r in range(len(row_cuts)):
            y1, y2 = y_lines[r], y_lines[r + 1]
            for x in row_cuts[r]:
                cv2.line(debug, (int(x), int(y1)), (int(x), int(y2)),
                         (0, 0, 255), 2)
    for y in y_lines:
        cv2.line(debug, (0, int(y)), (debug.shape[1] - 1, int(y)),
                 (0, 0, 255), 2)
    cv2.imwrite(str(filename), debug)


# ---------------------------------------------------------------------------
# Per-card detection – grid-line components
# ---------------------------------------------------------------------------

def find_card_components(image):
    """
    Locate each card as a textured region of the photo, then cut sheet
    blobs into individual cards.

    Line-morphology detection (opening with long kernels) fails on
    faintly-printed sheets: their grid lines break into short segments
    that no opening survives, so whole cards go missing. Cards are far
    more reliably characterised by texture: printed numbers make the
    local pixel std of the raw grey image jump, while table surface and
    shadows stay smooth. A smoothed std map therefore yields solid
    blobs per sheet. Those blobs are recursively halved along valleys
    of a print-projection (pixels notably darker than the local median,
    which ignores dark table pixels) until each piece is one card.

    Returns (labels, groups): an int32 label map where every accepted
    card has its own id, plus [{"ids": [id], "bbox": (x, y, w, h)}],
    unordered.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gf = gray.astype(np.float32)

    # Local texture energy: std of raw grey in a small window.
    mu = cv2.boxFilter(gf, -1, (31, 31))
    mu2 = cv2.boxFilter(gf * gf, -1, (31, 31))
    std = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    kk = max(31, (w // 10) | 1)
    sm = cv2.boxFilter(std, -1, (kk, kk))
    m = (sm > 18).astype(np.uint8) * 255
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, w // 30)))
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (w // 60, w // 60)))

    n_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(m, 8)

    # Print mask: clearly darker than the local median. On paper this is
    # ink; on dark table both reference and pixel are dark, so it stays
    # quiet — projections of this mask show card gutters as valleys even
    # when the gutter itself is dark table.
    locmed = cv2.medianBlur(gray, 51).astype(np.int16)
    print_m = (gray.astype(np.int16) < locmed - 35).astype(np.uint8) * 255

    # Pink mask: the saturated BINGO header colour. The header letters
    # are pink on pink — nearly invisible in grayscale — so the print
    # mask reads every header as a low-ink band and the projection
    # splitter mistakes it for a gutter, right where stacked cards
    # touch. The row projection below is therefore patched at
    # header-like pink bands (rows where pink is dense across the
    # piece); scattered pink elsewhere stays out of the way.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pink_m = (((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 162)) &
              (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 140)).astype(np.uint8) * 255

    def smooth1d(v, k):
        ker = np.ones(k, np.float32) / k
        return np.convolve(v.astype(np.float32), ker, mode="same")

    def cuts_1d(profile, min_gap, lo_frac=0.20):
        ref = np.percentile(profile, 75)
        low = profile < lo_frac * max(ref, 1e-6)
        out, x = [], 1
        while x < len(low) - 1:
            if low[x]:
                s = x
                while x + 1 < len(low) - 1 and low[x + 1]:
                    x += 1
                if x - s + 1 >= min_gap and s > 2 and x < len(low) - 3:
                    out.append((s, x))
            x += 1
        return out

    leaves = []

    def recurse(x, y, bw, bh, lid, depth):
        if bw < 0.10 * w or bh < 0.08 * h or depth > 5:
            if bw >= 0.10 * w and bh >= 0.08 * h:
                leaves.append([x, y, bw, bh, lid])
            return
        mm = blob_labels[y:y + bh, x:x + bw] == lid
        pm = np.where(mm, print_m[y:y + bh, x:x + bw], 0)
        colp = smooth1d(pm.sum(axis=0) / 255, max(5, w // 150))
        rowp = smooth1d(pm.sum(axis=1) / 255, max(5, w // 150))
        # Header bands of stacked cards read as gutters in rowp (pink
        # letters vanish from the print mask); raise those rows to the
        # profile's interquartile level so only true gutters stay low.
        pk = np.where(mm, pink_m[y:y + bh, x:x + bw], 0)
        pink_rows = smooth1d(pk.sum(axis=1) / 255.0, max(5, w // 150))
        band_rows = pink_rows > 0.15 * bw
        if band_rows.any():
            fill = float(np.percentile(rowp[~band_rows], 75)) \
                if (~band_rows).any() else float(rowp.max())
            rowp = np.where(band_rows, fill, rowp)
        min_gap = max(12, w // 100)
        ccols = cuts_1d(colp, min_gap)
        crows = cuts_1d(rowp, min_gap)
        cuts = ccols if len(ccols) >= len(crows) else crows
        axis = 0 if len(ccols) >= len(crows) else 1
        if not cuts:
            leaves.append([x, y, bw, bh, lid])
            return
        starts = [0] + [e + 1 for _, e in cuts]
        ends = [s for s, _ in cuts] + [(bw if axis == 0 else bh)]
        for a, b in zip(starts, ends):
            if b - a < (0.10 * w if axis == 0 else 0.08 * h):
                continue
            if axis == 0:
                recurse(x + a, y, b - a, bh, lid, depth + 1)
            else:
                recurse(x, y + a, bw, b - a, lid, depth + 1)

    for i in sorted(range(1, n_blobs),
                    key=lambda i_: -blob_stats[i_, cv2.CC_STAT_AREA]):
        x, y, bw, bh, area = blob_stats[i]
        if area < 0.005 * w * h:
            continue
        recurse(x, y, bw, bh, i, 0)

    # Stacked cards hide from the projection splitter: printed sheets
    # put consecutive rows of cards nearly in contact, so the seam has
    # no deep print valley — while inside a single card the gaps
    # between number rows do. The BINGO headers rescue this: every card
    # starts with a saturated pink band, so a leaf holding several
    # well-separated pink bands is several vertically stacked cards,
    # with each band marking a card's top edge. Photos without colour
    # headers have no pink at all and pass through untouched.

    def split_pink(leaf):
        x, y, bw, bh, lid = leaf
        rs = pink_m[y:y + bh, x:x + bw].sum(axis=1).astype(np.float32)
        k = max(15, bw // 100) | 1
        rs = np.convolve(rs, np.ones(k) / k, "same")
        if rs.max() < 40:
            return [leaf]
        hi = rs > max(0.30 * float(rs.max()), 40.0)
        peaks, i = [], 1
        while i < len(hi) - 1:
            if hi[i]:
                s = i
                while i + 1 < len(hi) - 1 and hi[i + 1]:
                    i += 1
                seg = rs[s:i + 1]
                peaks.append([s + int(seg.argmax()), float(seg.max())])
            i += 1
        # Pick card-start bands greedily from the strongest down: a
        # genuine stacked neighbour sits one card-height from an
        # accepted band, while pink digits inside the grid or the FREE
        # square at a card's centre sit mid-card, roughly half a pitch
        # from every header. Distance to the nearest accepted band lets
        # long stacks chain link by link. Title strips above the first
        # card fail the pitch test too, which simply leaves the crop
        # starting there.
        peaks = [(t, v) for t, v in peaks if t <= bh - max(150, int(0.12 * bh))]
        if not peaks:
            return [leaf]
        merged = [max(peaks, key=lambda p_: p_[1])]
        # A band rejected early can become acceptable once a nearer
        # header joins, so iterate until no new band qualifies.
        changed = True
        while changed:
            changed = False
            for p in sorted(peaks, key=lambda p_: -p_[1]):
                if p in merged:
                    continue
                d = min(abs(p[0] - a[0]) for a in merged)
                if 650 <= d <= 1500:
                    merged.append(p)
                    changed = True
        merged.sort(key=lambda p_: p_[0])
        if len(merged) < 2:
            return [leaf]
        segs, prev = [], 0
        for top, _ in merged:
            b = max(0, top - 12)  # card begins at its header
            if b - prev >= int(0.08 * h):
                segs.append((prev, b))
            prev = b
        if bh - prev >= int(0.08 * h):
            segs.append((prev, bh))
        return [[x, y + a, bw, b - a, lid] for a, b in segs] or [leaf]

    leaves = [piece for leaf in leaves for piece in split_pink(leaf)]

    # Trim each leaf down to its content: blob bboxes routinely include
    # stretches of bare table or neighbouring clutter, and a crop that
    # opens on table makes downstream row fitting draw its grid over
    # whatever is in the way. Rows/columns whose content (print or pink)
    # is negligible next to the leaf's typical content row are cut away.
    def trim_leaf(leaf):
        x, y, bw, bh, lid = leaf
        cont = np.where(blob_labels[y:y + bh, x:x + bw] == lid,
                        np.where((print_m > 0) | (pink_m > 0),
                                 np.uint8(255), np.uint8(0))[y:y + bh, x:x + bw],
                        0)

        def bounds(v):
            nz = v[v > 0]
            if not len(nz):
                return 0, len(v) - 1
            idx = np.nonzero(v >= 0.10 * np.median(nz))[0]
            return int(idx[0]), int(idx[-1])

        r0, r1 = bounds(cont.sum(axis=1) / 255.0)
        c0, c1 = bounds(cont.sum(axis=0) / 255.0)
        return [x + c0, y + r0, c1 - c0 + 1, r1 - r0 + 1, lid]

    leaves = [trim_leaf(l) for l in leaves]

    # Validation: plausible card shape, enough print ink, and size
    # comparable to the other cards of this photo (cards in one shot
    # share scale; stragglers are junk).
    out_labels = np.zeros(image.shape[:2], np.int32)
    groups = []
    nid = 1
    for x, y, bw, bh, lid in sorted(leaves, key=lambda l_: l_[1]):
        if bw < 0.12 * w or bh < 0.08 * h:
            continue
        ar = bw / bh
        if not 0.55 <= ar <= 1.9:
            continue
        mm = blob_labels[y:y + bh, x:x + bw] == lid
        pm = np.where(mm, print_m[y:y + bh, x:x + bw], 0)
        if pm.sum() / 255 < 0.05 * bw * bh:
            continue
        out_labels[y:y + bh, x:x + bw][mm] = nid
        groups.append({"ids": [nid], "bbox": (int(x), int(y), int(bw), int(bh))})
        nid += 1
    if groups:
        med_area = float(np.median(
            [g["bbox"][2] * g["bbox"][3] for g in groups]))
        keep = [g for g in groups
                if g["bbox"][2] * g["bbox"][3] >= 0.25 * med_area]
        if len(keep) != len(groups):
            bad = {g["ids"][0] for g in groups} - {g["ids"][0] for g in keep}
            for b in bad:
                out_labels[out_labels == b] = 0
            groups = keep
    return out_labels, groups


def extract_card_grids(image):
    """
    Warp each detected card component to a level crop of its number
    grid, ready for extract_grid_cells().

    The component mask only covers detected line pixels, and printed
    outer borders are often too faint to be part of it — cropping tight
    to the mask would cut through the outermost rows. So the warped
    crop is padded generously; extract_grid_cells' row fit locates the
    true boundaries within it.

    Returns (cards, labels, groups): one level crop per detected card,
    top-to-bottom, plus the component data the caller may want for
    debug overlays.
    """
    labels, groups = find_card_components(image)
    groups.sort(key=lambda g_: g_["bbox"][1])
    H, W = image.shape[:2]
    cards = []
    for grp in groups:
        gx, gy, gw, gh = grp["bbox"]
        # Generous window: the component mask routinely misses the
        # faintly-printed outermost grid borders, so the rect below
        # under-covers the true grid. Everything the fitter might need
        # must already be inside this window.
        pad_x = int(0.25 * gw)
        pad_y = int(0.40 * gh)
        x0, y0 = max(0, gx - pad_x), max(0, gy - pad_y)
        x1 = min(W, gx + gw + pad_x)
        y1 = min(H, gy + gh + pad_y)
        sub = image[y0:y1, x0:x1]
        submask = (
            np.isin(labels[y0:y1, x0:x1], grp["ids"]).astype(np.uint8) * 255
        )
        # A group split out of a fused component covers only part of its
        # members' pixels — restrict the mask to the group's x-span.
        if "xspan" in grp:
            xa, xb = grp["xspan"]
            lim = np.zeros(submask.shape[1], np.uint8)
            lim[max(0, xa - x0):max(0, xb - x0)] = 1
            submask &= lim[None, :] * 255

        rect = cv2.minAreaRect(
            np.column_stack(np.nonzero(submask))[:, ::-1].astype(np.float32)
        )
        box = order_points(cv2.boxPoints(rect))
        # Grow the quad about its centre BEFORE warping: four_point_warp
        # crops to the quad, so without this the canvas would hug the
        # mask exactly and any undetected grid border outside it would
        # be cropped away for good.
        centre = box.mean(axis=0)
        box = centre + (box - centre) * (1.12, 1.28)
        card = four_point_warp(sub, box)
        mask_w = four_point_warp(submask, box)

        mys, mxs = np.nonzero(mask_w > 127)
        ty, by = int(mys.min()), int(mys.max())
        tx, bx = int(mxs.min()), int(mxs.max())
        # A little extra margin beyond the detected line extent.
        pad_y = int(0.10 * (by - ty))
        pad_x = int(0.06 * (bx - tx))
        ty, tx = max(0, ty - pad_y), max(0, tx - pad_x)
        by = min(card.shape[0], by + pad_y)
        bx = min(card.shape[1], bx + pad_x)

        card = deskew(card[ty:by, tx:bx])
        cards.append(card)
    return cards, labels, groups

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(image_path, output_dir="output", number_of_cards=3):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    stem = image_path.stem

    # ---- 1. Detect cards ------------------------------------------------------
    # Preferred: find each card's grid directly (robust to strips whose
    # cards are tilted relative to each other, where no single strip
    # warp exists). Falls back to the legacy whole-strip pipeline when
    # no grid components can be found.
    try:
        card_images, labels, groups = extract_card_grids(image)
    except Exception as e:  # noqa: BLE001 - fall through to legacy path
        print(f"Per-card detection failed ({e}) – trying legacy strip pipeline")
        card_images = []

    if card_images:
        print(f"Detected {len(card_images)} card(s) via grid components")
        overview = image.copy()
        for grp in groups:
            gx, gy, gw, gh = grp["bbox"]
            cv2.rectangle(overview, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 8)
        cv2.imwrite(str(output_dir / f"{stem}_corners.png"), overview)
    else:
        sheet_corners, debug_mask = detect_sheet(
            image,
            debug_path=output_dir / f"{stem}_corners.png"
        )
        cv2.imwrite(str(output_dir / "debug_grid_lines.png"), debug_mask)

        if sheet_corners is None:
            raise RuntimeError("Could not detect the outer sheet boundary")

        save_corner_debug(image, sheet_corners,
                          output_dir / f"{stem}_corners_labelled.png")

        # ---- 2. Warp the whole strip -----------------------------------------
        warped_sheet = four_point_warp(image, sheet_corners)
        cv2.imwrite(str(output_dir / f"{stem}_warped_sheet.png"), warped_sheet)

        # ---- 3. Split into individual cards ----------------------------------
        cards = split_using_pink_headers(warped_sheet, number_of_cards)
        if cards is None or len(cards) != number_of_cards:
            print("Pink-header split failed – trying gap detection")
            cards = split_using_horizontal_gaps(warped_sheet, number_of_cards)
        if cards is None or len(cards) != number_of_cards:
            print("Gap detection failed – falling back to equal-height split")
            cards = split_sheet_into_cards(warped_sheet, number_of_cards)
        card_images = cards

    # ---- 4. Extract 5×5 cells from each card ---------------------------------
    for idx, card in enumerate(card_images):
        card_name = f"card_{idx + 1}"

        # Cancel any residual tilt — otherwise row-divider detection
        # smears and slices through digits. Done before saving so every
        # debug image and cell crop shares the same coordinates.
        angle = estimate_skew_angle(card)
        card = deskew(card)
        print(f"  {card_name}: deskewed by {angle:+.2f} deg")

        cv2.imwrite(str(output_dir / f"{card_name}.png"), card)

        cells, x_lines, y_lines, row_cuts = extract_grid_cells(card)
        q = grid_quality(card, x_lines, y_lines, row_cuts)
        print(f"  {card_name}: quality y_reg={q['y_reg']:.3f} "
              f"x_reg={q['x_reg']:.3f} ink={q['ink']:.4f}")
        if q["ink"] > 0.10 or q["y_reg"] > 0.30 or q["x_reg"] > 0.35:
            print(f"  {card_name}: REJECTED (grid does not fit a 5x5 card)")
            cv2.imwrite(str(output_dir / f"{card_name}_rejected.png"), card)
            continue
        save_cells(cells, output_dir / "cells", card_name)
        draw_grid_debug(card, x_lines, y_lines,
                        output_dir / f"{card_name}_grid_debug.png",
                        row_cuts=row_cuts)

        print(f"  {card_name}: extracted {len(cells)}×{len(cells[0])} cells")

    print(f"Done. Results written under: {output_dir.resolve()}")
    return card_images


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_file = "bingo_sheet.jpg"
    else:
        image_file = sys.argv[1]

    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    process_image(image_file, output_dir=out)
