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


RING_WEIGHT = 0.15     # ring-ink penalty weight in divider fitting
PRIOR_INJECT = True    # inject prior-consistent full-span models when nothing matches
PRIOR_MIN_STRENGTH = 0.30   # autocorr peak height (vs lag 0) to trust a prior
PRIOR_MIN_DOMINANCE = 2.0   # peak vs best rival lag — kills harmonic aliases
FIND_DEBUG = False     # print per-stage card-segmentation internals


def _ac_prior(profile, lo_frac=0.10, hi_frac=0.26):
    """
    Autocorrelation pitch estimate for a 1-D ink profile.

    Returns (lag, strength, dominance): the periodicity lag inside the
    plausible-pitch window, its height relative to lag 0, and its ratio
    against the strongest rival peak. A low dominance means the "pitch"
    is one indistinguishable bump among many — digit harmonics, plank
    grain, print noise — and must not be trusted as a hard constraint.
    """
    d = profile - profile.mean()
    n = len(profile)
    if n < 16:
        return None, 0.0, 0.0
    ac = np.correlate(d, d, mode="full")
    ac = ac[n - 1:]
    lo, hi = int(n * lo_frac), max(int(n * lo_frac) + 1, int(n * hi_frac))
    if hi <= lo or ac[0] <= 0:
        return None, 0.0, 0.0
    win = ac[lo:hi]
    pk = float(win.max())
    strength = pk / float(ac[0])
    if pk <= 0.05 * float(ac[0]):
        return None, strength, 1.0
    lag = int(np.argmax(win)) + lo
    m = int(0.25 * (lag - lo))
    mask = np.ones(len(win), bool)
    mask[max(0, lag - lo - m):min(len(win), lag - lo + m + 1)] = False
    alt = float(win[mask].max()) if mask.any() else 0.0
    return float(lag), strength, (pk / alt if alt > 0 else 99.0)


def _fit_regular_rows(candidates, h, rows, min_span_frac=0.45, max_span_frac=1.02,
                      miss_penalty=0.35, prior_pitch=None, prior_conf=None,
                      ring_profile=None):
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

    # No model agrees with the digit-pitch prior — either nothing fit
    # the crop-sized span box at all, or only aliased steps survived
    # (e.g. a card whose full span overflows the crop because a
    # physically overlapping neighbour or a shy mask hides part of it).
    # Trust the prior: add full-span models whose step matches it even
    # though their far boundary lies outside the image. The caller
    # clips such positions and the downstream quality gates judge what
    # remains — far better than letting an aliased half-pitch model win.
    if PRIOR_INJECT and prior_pitch and prior_conf \
            and prior_conf[0] >= PRIOR_MIN_STRENGTH \
            and prior_conf[1] >= PRIOR_MIN_DOMINANCE \
            and not any(abs(m[0] - prior_pitch) <= 0.16 * prior_pitch
                        for m in models):
        hi_span = 1.60 * h
        for cand in candidates:
            for a in range(n):
                top = cand[0] - a * prior_pitch
                bottom = top + (n - 1) * prior_pitch
                if min_span_frac * h <= bottom - top <= hi_span \
                        and -0.02 * h <= top <= h and bottom <= hi_span:
                    models.append((float(prior_pitch), cand[0], cand[0], a, a))

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
        # A regular fit can alias to half the true pitch, landing every
        # other boundary mid-digit. Real dividers sit in whitespace
        # gutters: ink in a ring around a boundary (excluding the line
        # itself) means that boundary slices through digit rows. Such
        # models lose against clean-gutter competitors even when they
        # harvest more line candidates.
        if ring_profile is not None:
            mu = float(ring_profile.mean())
            sd = float(ring_profile.std()) or 1.0
            inner = max(2, int(0.05 * step))
            # Clear the divider's own ink: printed lines run several rows
            # thick, and a narrow exclusion band would score the line's
            # shoulders against the model sitting on it.
            thick = [c[2] for c in candidates if len(c) > 2]
            if thick:
                inner = max(inner, int(round(0.75 * float(np.median(thick)))))
            outer = max(6, int(0.16 * step))
            for k in range(1, n - 1):
                p = int(round(positions[k]))
                lo_ = max(0, p - outer)
                hi_ = min(len(ring_profile), p + outer + 1)
                seg = np.concatenate([
                    ring_profile[lo_:max(lo_, p - inner)],
                    ring_profile[min(hi_, p + inner + 1):hi_],
                ])
                if len(seg):
                    score -= RING_WEIGHT * max((float(seg.mean()) - mu) / sd, 0.0)
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

    pitch, prior_strength, prior_dom = _ac_prior(sm, 0.08, 0.30)

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
                    raw.append((peak_x, float(vprof[peak_x]),
                                x - run_start + 1))
            x += 1

    # Merge hits from both scales (positions differ by a few px).
    raw.sort(key=lambda c: -c[1])
    lines = []
    for px, cov, th in raw:
        if all(abs(px - lx) > max(15, int(0.06 * w)) for lx, _, _ in lines):
            lines.append((px, cov, th))
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

    candidates = [(int(px), cov, th) for px, cov, th in lines]
    # Gutter dips are valleys, not printed strokes: nominal thin.
    candidates += [(int(dx), dip_strength, 3) for dx, _ in dips]
    candidates.sort()

    positions = _fit_regular_rows(candidates, w, columns, prior_pitch=pitch,
                                  prior_conf=(prior_strength, prior_dom),
                                  ring_profile=sm)
    if positions is None:
        return None

    # Guard against dividers drifting outside the card.
    return [int(np.clip(x, 0, w - 1)) for x in positions]


def _deskew_header(card):
    """
    Straighten a card whose header strip is visibly tilted inside an
    axis-aligned leaf crop. A line fitted through the saturated
    header's per-column centroids gives the residual angle the
    sheet-level warp leaves behind; rotating by it is what lets the
    header anchor place rows that stop slicing into digit bands
    (IMG_4903's sheet sits rotated counter-clockwise inside its leaf).
    Returns the card unchanged when there is no header or no usable
    tilt.
    """
    hdr = _find_header_band(card)
    if hdr is None:
        return card
    x0, x1, y0, y1 = hdr
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 100)).astype(np.uint8)
    win = sat[max(0, y0 - 10):min(y1 + 10, card.shape[0]), x0:x1 + 1]
    wsum = win.sum(axis=0).astype(np.float32)
    good = wsum >= 3
    if int(good.sum()) < 30:
        return card
    xs = np.nonzero(good)[0].astype(np.float32)
    cy = (win * np.arange(win.shape[0])[:, None]).sum(axis=0)[good] \
        / wsum[good]
    slope, _ = np.polyfit(xs, cy, 1)
    ang = float(np.degrees(np.arctan(slope)))
    if abs(ang) < 0.4 or abs(ang) > 10:
        return card
    h, w = card.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(card, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def _find_header_band(warped):
    """
    Locate the B I N G O header strip of a card crop.

    Every card opens with a full-width header band — same letters, same
    size, only the background colour changes — which makes it a far more
    reliable anchor than the grid's faint printed lines. Detection is
    colour-generic: a horizontal strip near the crop top whose pixels
    are mostly saturated (any hue), spanning a contiguous run of at
    least a third of the crop width.

    Returns (x0, x1, y0, y1), or None when no header-like band exists.
    """
    h, w = warped.shape[:2]
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 100)).astype(np.uint8)
    kk = max(3, h // 60)
    rows = np.convolve(sat.sum(axis=1).astype(np.float32) / w,
                       np.ones(kk) / kk, "same")
    hi = rows > 0.45
    min_h = max(5, int(0.025 * h))
    max_h = int(0.22 * h)
    y = 1
    while y < len(hi) - 1:
        if hi[y]:
            s = y
            while y + 1 < len(hi) - 1 and hi[y + 1]:
                y += 1
            if s > 0.40 * h:
                break  # headers live at the crop top; later bands are
                       # coloured squares inside the grid (e.g. FREE)
            if min_h <= y - s + 1 <= max_h:
                seg = sat[s:y + 1, :]
                cols = np.convolve(
                    seg.sum(axis=0).astype(np.float32) / (y - s + 1),
                    np.ones(9) / 9, "same")
                xs = np.nonzero(cols > 0.45)[0]
                if len(xs):
                    # Merge across pale letters: one faded glyph must
                    # not truncate the span (it sets the grid pitch).
                    gap_tol = max(25, int(0.06 * w))
                    runs, st = [], int(xs[0])
                    for a, b in zip(xs, xs[1:]):
                        if b - a > gap_tol:
                            runs.append((st, int(a)))
                            st = int(b)
                    runs.append((st, int(xs[-1])))
                    x0, x1 = max(runs, key=lambda r: r[1] - r[0])
                    if x1 - x0 >= 0.35 * w:
                        return x0, x1, s, y
        y += 1
    return None


def _anchor_grid_from_header(card_shape, hdr, row_ink,
                             prior_pitch=None, prior_conf=None):
    """
    Derive the six row boundaries of a card's number grid from its
    header band. The header spans the card's full width and sits flush
    above the grid, so its width predicts the row pitch: across the
    corpus the grid measures ~0.84 of the header width in height, i.e.
    five rows of ~0.168 header widths each. Expected boundaries are
    snapped to the nearest whitespace valley of the row-ink profile so
    cuts stay out of printed lines and digits.

    Returns 6 y positions, or None when the geometry cannot fit.
    """
    h = card_shape[0]
    x0, x1, _, hy1 = hdr
    hdr_w = x1 - x0
    top = hy1

    # Whitespace level: valleys between digit bands sit well below the
    # ink inside them. Snap each expected boundary to the CENTRE of the
    # low-ink run nearest it — centres are stable, argmin drifts around
    # flat valleys and inflates the regularity metric.
    quiet = 0.35 * float(np.median(row_ink[row_ink > 0])) if (row_ink > 0).any() \
        else 0.35 * float(row_ink.max())
    kk = max(3, int(0.04 * max(24.0, 0.168 * hdr_w))) | 1
    sm_ink = np.convolve(row_ink, np.ones(kk) / kk, mode="same")
    low = row_ink <= max(quiet, 1e-6)
    runs = []
    y = 0
    n = len(low)
    while y < n:
        if low[y]:
            s = y
            while y + 1 < n and low[y + 1]:
                y += 1
            runs.append((s, y))
        y += 1

    def snap(expected, pitch):
        best, best_d = None, None
        for s, e in runs:
            c = 0.5 * (s + e)
            d = abs(c - expected)
            if d <= 0.18 * pitch and (best_d is None or d < best_d):
                best, best_d = c, d
        return (expected if best is None else best), (best_d is None)

    # Candidate pitches, tried best-scored first:
    #   geom — width rule (~0.168 header widths per row); right for
    #          square-ish cells but too tall whenever pale letters
    #          shrink the measured span or cells are wide-format.
    #   cal  — the candidate refined onto the deepest whitespace valley
    #          near its own first row gap (ground truth for one step).
    #   squeeze — five rows divided into whatever space the crop offers
    #          below the header; rescues perspective-shrunk cards whose
    #          true grid simply cannot fit the width rule (IMG_4903
    #          card 1). Only kept when its valleys check out.
    p_geom = float(np.clip(0.168 * hdr_w, 24, h))
    if prior_pitch and prior_conf \
            and prior_conf[0] >= PRIOR_MIN_STRENGTH \
            and prior_conf[1] >= PRIOR_MIN_DOMINANCE \
            and abs(prior_pitch - p_geom) <= 0.18 * p_geom:
        p_geom = float(prior_pitch)

    def calibrate(p0):
        lo_, hi_ = int(top + 0.55 * p0), min(len(sm_ink) - 1,
                                             int(top + 1.45 * p0))
        if hi_ - lo_ < 8:
            return None
        v = lo_ + int(np.argmin(sm_ink[lo_:hi_ + 1]))
        p_new = float(v - top)
        ref = float(np.percentile(sm_ink[max(0, int(top)):hi_ + 1], 20))
        if 0.72 * p0 <= p_new <= 1.35 * p0 \
                and sm_ink[v] <= 1.3 * max(ref, 1e-6):
            return p_new
        return None

    cands = [p_geom]
    for base in (p_geom,
                 (h - 4 - top) / 5.0 * 0.96):
        pc = calibrate(base)
        if pc is not None and all(abs(pc - c) > 0.06 * c for c in cands):
            cands.append(pc)
    squeeze = (h - 4 - top) / 5.0 * 0.96
    if squeeze < 0.99 * p_geom:
        cands.append(squeeze)

    best = None
    for pitch in sorted(set(round(c, 1) for c in cands), reverse=True):
        positions = [float(top)]
        overflow = False
        score = 0.0
        for k in range(1, 6):
            expected = top + k * pitch
            if expected >= h + 0.10 * pitch:
                overflow = True  # grid runs past the crop: wrong pitch
                break
            pos, missed = snap(expected, pitch)
            if pos <= positions[-1]:
                pos = positions[-1] + max(4.0, 0.25 * pitch)
            vi = min(n - 1, max(0, int(pos)))
            depth = float(sm_ink[vi]) / max(
                float(np.percentile(sm_ink[max(0, int(top)):n], 50)), 1e-6)
            score += (3.0 if missed else min(abs(pos - expected), 1.5 *
                                             0.18 * pitch) /
                      (0.18 * pitch)) + depth
            positions.append(pos)
        if overflow:
            continue
        score /= 5.0
        if best is None or score < best[0]:
            best = (score, positions)
    if best is None:
        return None
    positions = best[1]
    if positions[-1] > h + 0.05 * p_geom:
        positions[-1] = float(h - 1)
    return [int(round(p)) for p in positions]


def _band_letter_census(binary, y0, y1):
    """
    Count components in a row band, split into total vs 'big' ones
    (taller than 55% of the band). Digit rows carry many tall glyphs;
    a B I N G O letter strip carries only a few wide flat ones.
    """
    # Degenerate bands (boundaries collapsed by aggressive snapping)
    # must not reach connectedComponentsWithStats — an empty image
    # segfaults it.
    h = binary.shape[0]
    a, b = max(0, int(y0)), min(h, max(int(y1), int(y0) + 1))
    if b <= a:
        return 0, 0
    band = np.ascontiguousarray(binary[a:b])
    n, _, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    bh = max(1, b - a)
    comps = [(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30]
    big = sum(1 for _, ch in comps if ch > 0.55 * bh)
    return len(comps), big


def _fit_column_lattice(gray, hdr, columns, y_lines):
    """
    Fit the vertical divider lattice to CONTENT rather than assuming the
    number grid spans exactly the header strip's width. The header can
    overhang the grid (title letters, FREE-square column, perspective),
    and a blind linspace(hdr_x0, hdr_x1) then drifts every interior cut
    into the neighbouring cell, splitting two-digit numbers
    (IMG_4903 card 2: true pitch varies 190-271 px while linspace
    assumed a constant 247).

    Grid search around the header-implied model: pitch in [0.78,
    1.15]x(hdr_w/columns), start shifted within about a fifth of a
    pitch. Scored by print-mask ink AT the interior cut positions
    (lower = cuts run through whitespace gutters). Marked/daubed cards
    smear ink across gutters, so the profile averages only the least-
    marked grid rows. Returns None when no candidate meaningfully beats
    the header-implied span, letting callers fall back to it.
    """
    h_, w_ = gray.shape[:2]
    lm = cv2.medianBlur(gray, 51).astype(np.int16)
    pmc = gray.astype(np.int16) < lm - 35

    # Per-row profiles; keep the cleanest half so daubs and stray
    # handwriting cannot dictate the lattice.
    row_profs = []
    for r in range(len(y_lines) - 1):
        a = int(np.clip(y_lines[r] + (y_lines[r + 1] - y_lines[r]) // 6,
                        0, h_))
        b = int(np.clip(y_lines[r + 1] - (y_lines[r + 1] - y_lines[r]) // 6,
                        0, h_))
        if b - a < 12:
            continue
        pr = pmc[a:b, :].mean(axis=0)
        row_profs.append((float(pr.mean()), pr))
    if not row_profs:
        return None
    row_profs.sort(key=lambda t_: t_[0])
    keep = max(1, len(row_profs) // 2)
    kept = [p_ for _, p_ in row_profs[:keep]]
    prof = np.mean(kept, axis=0)
    smo = np.convolve(prof, np.ones(9) / 9, mode="same")

    # Cross-row gutter support: a TRUE cell boundary is blank in every
    # clean row, while the gap between the two digits of a number is
    # blank only in rows whose numbers happen to leave space there.
    # Ink-at-cut alone cannot tell the two apart (both are whitespace),
    # which is exactly how cuts ended up slicing 2-digit numbers in
    # half. A candidate position is valid only when most clean rows
    # agree it is clear.
    row_smos = [np.convolve(p_, np.ones(7) / 7, mode="same") for p_ in kept]
    thresh = [max(0.05, float(np.percentile(rs, 25))) for rs in row_smos]

    def support(x, win=0):
        xi = int(np.clip(x, 0, w_ - 1))
        lo = max(0, xi - win)
        hi = min(w_ - 1, xi + win)
        ok = sum(1 for rs, th in zip(row_smos, thresh)
                 if float(rs[lo:hi + 1].min()) <= th + 0.01)
        return ok / len(row_smos)

    # ----- Number-start periodicity anchored search -------------------------
    # The header strip can be unreadable on damaged crops (pink cut
    # residue, torn edge), and then hdr anchors the whole lattice to the
    # wrong span: a phantom left-edge column appears, every cut lands
    # between digit pairs, and column 5 falls off the right side
    # (IMG_4949 cards 2/4/5/6). Numbers are laid out on a strict pitch,
    # so the STARTS of digit blobs are periodic modulo the true column
    # pitch even when individual rows are smeared by handwriting or
    # daubs. Suppress the printed rules (they glue a whole row into one
    # component), collect digit-blob starts, find the pitch that
    # maximises their wrapped circular concentration, then refine phase
    # and pitch by minimising ink overlap between candidate cuts and
    # blobs.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   (max(30, w_ // 5), 1))
    rule = cv2.dilate(cv2.morphologyEx(pmc.astype(np.uint8),
                                       cv2.MORPH_OPEN, hk),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    clean = pmc.astype(np.uint8) & ~rule
    ivs = []
    for r in range(len(y_lines) - 1):
        a = int(np.clip(y_lines[r] + (y_lines[r + 1] - y_lines[r]) // 6,
                        0, h_))
        b = int(np.clip(y_lines[r + 1] - (y_lines[r + 1] - y_lines[r]) // 6,
                        0, h_))
        if b - a < 12:
            continue
        nl, _, stats, _ = cv2.connectedComponentsWithStats(clean[a:b, :], 8)
        for sx, sy, sw, sh, ar in stats[1:]:
            if (ar >= 20 and sh >= max(9, 0.30 * (b - a))
                    and sw >= 13 and sw < int(0.30 * w_)):
                ivs.append((int(sx), int(sx + sw)))
    peaks = []
    if len(ivs) >= columns * 3:
        starts = np.array([s for s, _ in ivs], float)
        wts = np.array([e - s for s, e in ivs], float)
        raw = []
        for p in np.arange(110.0, min(260.0, w_ / 3), 0.5):
            r_ = abs((np.exp(2j * np.pi * starts / p) * wts).mean())
            raw.append((float(r_), float(p)))
        raw.sort(reverse=True)
        for r_, p in raw:
            if all(abs(p - q) > 6 for q, _ in peaks):
                peaks.append((r_, p))
            if len(peaks) >= 3:
                break

        span_lo = min(s for s, _ in ivs)
        span_hi = max(e for _, e in ivs)

        def pen(b0, p, tol=7):
            tot = 0.0
            for s, e in ivs:
                for k in range(1, columns):
                    x = b0 + k * p
                    ov = min(e, x + tol) - max(float(s), x - tol)
                    if ov > 0:
                        tot += ov
            edge = 0.18 * p
            for k in range(columns + 1):
                x = b0 + k * p
                if x < span_lo - edge:
                    tot += 3 * (span_lo - edge - x)
                if x > span_hi + edge:
                    tot += 3 * (x - span_hi - edge)
            return tot

        best = None
        for _, p_pk in peaks:
            ang = np.exp(2j * np.pi * starts / p_pk)
            phi = float(np.angle((ang * wts).mean())) * p_pk / (2 * np.pi)
            phi %= p_pk
            inset = float(np.median((starts - phi) % p_pk))
            b00 = phi - inset
            for p in np.arange(p_pk - 4, p_pk + 4.01, 0.25):
                for db in np.arange(-40, 40.01, 1.0):
                    v = pen(b00 + db, p)
                    if best is None or v < best[0]:
                        best = (v, b00 + db, p)
        if best is not None:
            v, b0, p = best
            # Reject the fit when blobs disagree badly with ANY lattice:
            # a card whose numbers are not remotely gridded would get a
            # forced nonsense emission here otherwise. Normalised against
            # blob count — a handful of pixels of overlap per blob is
            # normal handwriting noise.
            if v <= max(80.0, 3.5 * len(ivs)):
                out = [int(round(b0 + k * p)) for k in range(columns + 1)]
                if out[0] >= 0 and out[-1] <= w_ - 1:
                    return out
        # No trustworthy periodicity — fall through to the ink/support
        # path below rather than forcing a bad fit.

    x0_hdr = float(hdr[0])
    p0 = (hdr[1] - hdr[0]) / float(columns)
    if p0 < 12:
        return None

    best = None
    lo_lim = x0_hdr - 0.25 * p0
    hi_lim = float(hdr[1]) + 0.15 * p0

    def in_margin(x):
        # A cut sits in a margin when its clear run is far wider than
        # any inter-column gutter (~0.15 pitch): blank paper beyond the
        # grid scores zero ink and would lure the search outside the
        # card entirely.
        if smo[int(np.clip(x, 0, w_ - 1))] > 0.05:
            return False
        l_ = r_ = int(x)
        while l_ - 1 >= 0 and smo[l_ - 1] <= 0.05:
            l_ -= 1
        while r_ + 1 < w_ and smo[r_ + 1] <= 0.05:
            r_ += 1
        return (r_ - l_) > 0.45 * p0

    fs = 0.78
    while fs <= 1.151:
        p = p0 * fs
        dstep = max(2, int(0.02 * p))
        dx = int(-0.18 * p)
        while dx <= int(0.12 * p):
            xs = [int(round(x0_hdr + dx + k * p)) for k in range(1, columns)]
            # Keep every cut inside the plausible grid span AND out of
            # page margins, and require cross-row support so digit gaps
            # cannot masquerade as cell boundaries. Keystone tilt shifts
            # a boundary a few px between rows, hence the +-window and
            # mean-based (not all-or-nothing) support test.
            sup_win = max(4, int(0.03 * p))
            sups = [support(x, sup_win) if
                    (lo_lim <= x <= hi_lim and 3 <= x <= w_ - 4
                     and not in_margin(x)) else 0.0 for x in xs]
            if float(np.mean(sups)) >= 0.7:
                s = sum(smo[x] for x in xs) / len(xs)
                # mild preference to stay near the header-implied model
                x_end = x0_hdr + dx + columns * p
                s += 0.004 * ((abs(dx) + abs(x_end - hdr[1])) / p)
                if best is None or s < best[0]:
                    best = (s, [int(round(x0_hdr + dx + k * p))
                                for k in range(columns + 1)])
            dx += dstep
        fs += 0.02

    if best is None:
        return None
    # Only leave the header-implied span when content clearly prefers
    # it: a marginal win just perturbs healthy grids and trips the
    # regularity checks on borderline crops. But when the header span
    # itself has no cross-row gutter support (phantom left margin, e.g.
    # IMG_4949 cards 2/5/6), a supported candidate always wins.
    base_xs = [x0_hdr + k * p0 for k in range(1, columns)]
    if all(3 <= x <= w_ - 4 for x in base_xs):
        base_s = sum(smo[int(round(x))] for x in base_xs) / len(base_xs)
        sup_win = max(4, int(0.03 * p0))
        base_sup = float(np.mean([support(x, sup_win)
                                  for x in base_xs]))
        if base_sup >= 0.7 and best[0] > base_s * 0.85:
            return None
    # Emit a UNIFORM lattice from the winning start/pitch rather than the
    # raw per-cut positions: the collision snap below localises each cut,
    # while downstream regularity checks expect even spacing (a ragged
    # global lattice pushed borderline cards past the reject threshold).
    xs = best[1]
    x_start = float(xs[0])
    p_fit = (float(xs[-1]) - x_start) / float(columns)
    return [int(round(x_start + k * p_fit)) for k in range(columns + 1)]


def extract_grid_cells(warped, rows=5, columns=5, min_junk_pitch=0):
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
    edge_z = max(8, int(0.02 * h))
    candidates = []
    y = 0
    while y < h:
        if hprof[y] > floor * w:
            run_start = y
            while y + 1 < h and hprof[y + 1] > floor * w:
                y += 1
            # A generous edge zone rejects deskew slivers AND the dark
            # paper/table transition, which on tight crops sits far
            # enough inside to read as a strong full-width divider
            # (IMG_4956 card 5: it dragged the whole row fit one band
            # low, silently discarding the top row of numbers).
            touches_edge = run_start <= edge_z or y >= h - edge_z
            too_thick = (y - run_start + 1) > max_run
            if not touches_edge and not too_thick:
                run = hprof[run_start:y + 1]
                peak_y = run_start + int(np.argmax(run))
                candidates.append((peak_y, float(run.max()) / w,
                                   y - run_start + 1))
        y += 1

    # Digit-ink banding gives an independent read of the row pitch:
    # digit bands repeat with the row period even where every printed
    # divider is too faint to detect. Fed to the fitter as a prior, it
    # rules out aliased step sizes when only sparse dividers survive.
    ink = binary.mean(axis=1).astype(np.float32) / 255.0
    ink = np.convolve(ink, np.ones(15) / 15, mode="same")
    prior_pitch, prior_strength, prior_dom = _ac_prior(ink, 0.10, 0.26)
    prior_conf = (prior_strength, prior_dom)

    # ----- Header-anchored grid ----------------------------------------------
    # The B I N G O strip pins the grid geometry without any divider
    # evidence: its bottom edge is the grid top and its width predicts
    # the row pitch. Tried first; the line fitter below remains the
    # path for headerless sheets.
    hdr = _find_header_band(warped)
    fitted = None
    if hdr is not None:
        if FIND_DEBUG:
            print(f"  [grid] header band x{hdr[0]}-{hdr[1]} y{hdr[2]}-{hdr[3]}")
        fitted = _anchor_grid_from_header(warped.shape, hdr, ink,
                                          prior_pitch=prior_pitch,
                                          prior_conf=prior_conf)

    if fitted is None:
        fitted = _fit_regular_rows(candidates, h, rows, prior_pitch=prior_pitch,
                                   prior_conf=prior_conf,
                                   ring_profile=ink)
    rows_fitted = fitted is not None

    # A line-fitted grid can still open on the header strip: its bottom
    # border reads as a strong divider, so the letters land in "row 1"
    # and the true last row falls past the last boundary. The letter
    # census exposes this — a header row holds a handful of wide flat
    # glyphs where a digit row holds many tall ones — and the fix is to
    # drop the first boundary and extrapolate one pitch at the bottom.
    if rows_fitted and hdr is None:
        n_tot, big0 = _band_letter_census(binary, fitted[0] + 4,
                                          fitted[1] - 4)
        others_big = []
        for k in range(1, min(rows, 4) + 1):
            _, bk = _band_letter_census(binary, fitted[k] + 4,
                                        fitted[k + 1] - 4)
            others_big.append(bk)
        if n_tot >= 3 and big0 <= 5 and others_big \
                and float(np.median(others_big)) >= 10:
            step = float(np.median(np.diff(fitted)))
            shifted = list(fitted[1:]) + [fitted[-1] + step]
            if FIND_DEBUG:
                print(f"  [grid] header-as-row-1 detected (big0={big0}, "
                      f"others={others_big}); shifting grid down one pitch")
            fitted = [int(round(v)) for v in shifted]
            rows_fitted = True

    # Coverage repair: a line-fitted grid can also START one band low —
    # sparse divider evidence plus a paper-edge artefact leaves the top
    # row of numbers above fitted[0] entirely (IMG_4956 card 5: grid
    # began on row 2). When the band just above the fitted top carries
    # digit-row glyph density, shift the whole lattice up one pitch and
    # drop the trailing boundary.
    if rows_fitted and hdr is None:
        step0 = float(np.median(np.diff(fitted)))
        up_y0 = int(fitted[0] - step0)
        if up_y0 >= int(0.05 * h):
            _, big_up = _band_letter_census(binary, up_y0 + 4,
                                            fitted[0] - 4)
            # Only shift when the space BELOW the grid is empty margin
            # (paper-edge artefact pulled the fit down). If rows continue
            # below, the band above is more likely a title strip and the
            # fit was right all along.
            below_a = int(min(h - 1, fitted[-1] + 4))
            below_b = int(min(h, fitted[-1] + max(6, int(0.6 * step0))))
            ink_below = float(binary[below_a:below_b].mean() / 255.0) \
                if below_b - below_a >= 4 else 1.0
            if big_up >= 8 and ink_below < 0.02:
                if FIND_DEBUG:
                    print(f"  [grid] row missing above fitted top "
                          f"(big_up={big_up}); shifting grid up one pitch")
                fitted = ([up_y0] + list(fitted[:-1]))
                rows_fitted = True

    y_lines = fitted

    if y_lines is None:
        # No divider evidence at all — bare table yields no candidates.
        # Callers treat rows_fitted=False as grounds for rejection.
        y_lines = list(np.linspace(int(h * 0.12), int(h * 0.92), rows + 1).astype(int))
    y_lines = [int(np.clip(y, 0, h - 1)) for y in y_lines]

    # ----- Vertical dividers -------------------------------------------------
    # Primary: printed-line + gutter-dip evidence fitted with a
    # regular-spacing model (_fit_vertical_dividers). Digit-centroid
    # clustering below is only a fallback: its component filters reject
    # many real digits, which starves the k-means of clusters and lets it
    # invent dividers that slice straight through numbers.
    # When a header anchored the rows, its x-span brackets the card:
    # the six dividers simply tile that span, no line evidence needed —
    # these sheets' verticals are as faint as their horizontals, and
    # neighbouring cards' columns can leak in through fused crops.
    # The collision-escape pass below cleans up any residual drift.
    y0 = max(0, y_lines[0] - 5)
    y1 = min(h, y_lines[-1] + 5)
    x_shift = 0
    if hdr is not None:
        x_lines = _fit_column_lattice(gray, hdr, columns, y_lines)
        if x_lines is None:
            x_lines = [int(round(v)) for v in
                       np.linspace(hdr[0], hdr[1], columns + 1)]
        # Lattice search can push an endpoint past the crop edge
        # (oversized pitch); clamp so downstream window bounds stay
        # inside the profile arrays.
        x_lines = [int(np.clip(v, 2, w - 2)) for v in x_lines]
    else:
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
    # another. Each cut therefore snaps per row to the least-ink spot
    # within a NARROW symmetric window around its fitted position.
    # Bounded drift preserves cell widths (an earlier corridor-chasing
    # version let cuts wander half a column, producing razor-thin and
    # double-wide cells), and the least-ink target still works on
    # daubed/marked rows where no true whitespace corridor exists.
    step = float(np.median(np.diff(x_lines)))
    min_sep = max(8, int(0.05 * step))
    wx = max(3, int(0.20 * step))  # half-window for vertical cut snaps

    # Full-crop print mask, shared by the vertical and the per-column
    # horizontal refinement below. Local-median, not a fixed threshold:
    # flat shadows must not read as ink.
    locmed_full = cv2.medianBlur(gray, 51).astype(np.int16)
    pm_full = gray.astype(np.int16) < locmed_full - 35

    row_cuts = []
    for r in range(rows):
        y_top = int(np.clip(y_lines[r] + (y_lines[r + 1] - y_lines[r]) // 8, 0, h))
        y_bot = int(np.clip(y_lines[r + 1] - (y_lines[r + 1] - y_lines[r]) // 8, 0, h))
        if y_bot - y_top < 12:
            row_cuts.append(list(x_lines))
            continue
        band = pm_full[y_top:y_bot].mean(axis=0)
        sm_band = np.convolve(band, np.ones(7) / 7, mode="same")

        def run_centre(cand, a, b):
            # Gutters between digit groups read as plateaus of low ink;
            # plain argmin lands anywhere on the plateau — often on its
            # edge, splitting the neighbouring cell's digits. Expand to
            # the contiguous near-minimum run and take its midpoint so
            # the cut sits in the middle of the whitespace.
            t = sm_band[cand] + 0.015
            l_ = r_ = cand
            while l_ - 1 >= a and sm_band[l_ - 1] <= t:
                l_ -= 1
            while r_ + 1 < b and sm_band[r_ + 1] <= t:
                r_ += 1
            return (l_ + r_) // 2

        cuts = []
        for k, xp in enumerate(x_lines):
            xp = int(np.clip(xp, 0, w - 1))
            lo = cuts[k - 1] + min_sep if k else 2
            hi = int(x_lines[k + 1]) - min_sep if k + 1 < len(x_lines) else w - 3
            a, b = max(lo, xp - wx), min(hi + 1, xp + wx + 1)
            pos = xp
            if b - a > 6:
                cand = run_centre(a + int(np.argmin(sm_band[a:b])), a, b)
                # Move only for a real improvement: on fully daubed rows
                # every candidate sits in ink, and chasing the least-ink
                # pixel just makes cuts wander off the printed grid.
                if sm_band[cand] <= 0.05 or \
                        sm_band[cand] < sm_band[xp] - 0.01:
                    pos = cand
                elif sm_band[pos] > 0.10:
                    # Still colliding: keystone can push a gutter past
                    # the standard window on edge rows. One wider,
                    # clamped retry before giving up.
                    wa = max(lo, xp - 2 * wx)
                    wb = min(hi + 1, xp + 2 * wx + 1)
                    if wb - wa > 6:
                        wc = run_centre(wa + int(np.argmin(sm_band[wa:wb])),
                                        wa, wb)
                        if sm_band[wc] < sm_band[pos] - 0.02:
                            pos = wc
            pos = int(np.clip(pos, lo, hi))
            cuts.append(pos)
        for k in range(1, len(cuts)):
            cuts[k] = max(cuts[k], cuts[k - 1] + min_sep)
        cuts[-1] = min(cuts[-1], w - 1)
        row_cuts.append(cuts)

    # Tilt-aware vertical smoothing: the same keystone drifts column
    # boundaries across card height (IMG_4948 card 3: bottom two rows
    # sliced). Fit each interior cut with a quadratic trend over row
    # centres (perspective bends dividers, a line cannot track that —
    # IMG_4956 card 2) and clamp residuals.
    vstep = float(np.median(np.diff(y_lines)))
    row_ycent = [0.5 * (y_lines[r] + y_lines[r + 1]) for r in range(rows)]
    clamp_x = max(5, int(0.22 * float(np.median(np.diff(x_lines)))))
    for k in range(1, len(row_cuts[0]) - 1):
        pxs = [row_ycent[r] for r in range(rows)
               if len(row_cuts[r]) == len(row_cuts[0])]
        pys = [row_cuts[r][k] for r in range(rows)
               if len(row_cuts[r]) == len(row_cuts[0])]
        if len(pxs) >= 3 and float(np.ptp(pxs)) > 1:
            z = np.polyfit(pxs, pys, 2 if len(pxs) >= 5 else 1)
            # One row's escape measurement can be dragged into a
            # scribble (bottom rows sit in shadow/clipping) and bend
            # the shared trend for everyone. Drop the single worst
            # residual — if it is truly extreme — and refit.
            if len(pxs) >= 5:
                resids = [abs(float(np.polyval(z, pxs[i])) - pys[i])
                          for i in range(len(pxs))]
                worst = int(np.argmax(resids))
                lim = 0.12 * float(np.median(np.diff(x_lines)))
                if resids[worst] > lim:
                    keep = [i for i in range(len(pxs)) if i != worst]
                    z = np.polyfit([pxs[i] for i in keep],
                                   [pys[i] for i in keep],
                                   2 if len(keep) >= 4 else 1)
            for r in range(rows):
                if len(row_cuts[r]) != len(row_cuts[0]):
                    continue
                pred = float(np.polyval(z, row_ycent[r]))
                dev = row_cuts[r][k] - pred
                row_cuts[r][k] = int(round(
                    pred + float(np.clip(dev, -clamp_x, clamp_x))))
            # restore monotonicity after clamping
            for r in range(rows):
                if len(row_cuts[r]) != len(row_cuts[0]):
                    continue
                min_sep_k = max(4, int(0.10 *
                                       float(np.median(np.diff(x_lines)))))
                for kk in range(1, len(row_cuts[r])):
                    row_cuts[r][kk] = max(row_cuts[r][kk],
                                          row_cuts[r][kk - 1] + min_sep_k)

    # ----- Row shear model ----------------------------------------------------
    # Keystone tilts the whole row structure: boundaries descend (or
    # rise) linearly across the card width, so a single global y_lines
    # is only correct at one horizontal position. Independent per-
    # column snapping cannot recover it — on marked cards the least-ink
    # spot inside a column is usually a DIGIT GAP, not the boundary
    # (IMG_4903 card 2: greens sank into numbers toward the right,
    # stacking two numbers into one cell). Printed rules are too faint
    # to track here, so estimate ONE global shear from the ink itself:
    # the correct slope maximises the contrast of the row projection
    # after de-shearing, because digit bands align into sharp stripes.
    # A straight shear cannot express keystone curvature (rows converge
    # toward one side when the card sat off-axis: IMG_4948 cards 2/4/5
    # and IMG_4949 card 6, worse on one edge), so after the slope a
    # quadratic term q*(u^2 - 1/3) is fitted the same way — u normalised
    # to [-1,1] across the grid, -1/3 centring keeps the mean shift at
    # zero so fitting q never drags every boundary up or down together.
    ystep = float(np.median(np.diff(y_lines)))
    gx0 = int(np.clip(x_lines[0], 0, w - 2))
    gx1 = int(np.clip(x_lines[-1] + 1, 1, w))
    gy0 = int(np.clip(y_lines[0], 0, h - 2))
    gy1 = int(np.clip(y_lines[-1] + 1, 1, h))
    sub = pm_full[gy0:gy1, gx0:gx1]
    gys, gxs = np.nonzero(sub)
    shear = 0.0
    quad = 0.0
    cube = 0.0
    xc = float(0.5 * (x_lines[0] + x_lines[-1]))
    hw = max(1.0, 0.5 * (x_lines[-1] - x_lines[0]))
    if len(gys) > 500:
        if len(gys) > 300000:  # subsample for speed; statistics suffice
            idx = np.random.default_rng(7).choice(len(gys), 300000,
                                                  replace=False)
            gys, gxs = gys[idx], gxs[idx]
        xs_f = gxs.astype(np.float32) + gx0
        ys_f = gys.astype(np.float32) + gy0
        us_f = ((xs_f - xc) / hw).astype(np.float32)
        best_v = -1.0

        def shear_score(s, qv=0.0, cv_=0.0):
            yy = np.clip(ys_f + s * (xs_f - xc)
                         + qv * (us_f * us_f - 1.0 / 3.0)
                         + cv_ * (us_f * us_f * us_f), 0, h - 1)
            prof = np.bincount(yy.astype(np.int32), minlength=h)
            prof = np.convolve(prof.astype(np.float32),
                               np.ones(9) / 9, mode="same")
            return float(np.var(prof))

        coarse = np.arange(-0.08, 0.0801, 0.004)
        s0 = max(coarse, key=shear_score)
        fine = np.arange(s0 - 0.004, s0 + 0.00401, 0.001)
        shear = float(max(fine, key=shear_score))
        if abs(shear) < 0.005:
            shear = 0.0  # noise level; keep global lines

        cquad = np.arange(-60, 60.1, 5)
        q0 = max(cquad, key=lambda v_: shear_score(shear, v_))
        fquad = np.arange(q0 - 5, q0 + 5.01, 1)
        quad = float(max(fquad, key=lambda v_: shear_score(shear, v_)))
        if abs(quad) < 4:
            quad = 0.0  # below noise; straight boundaries are fine

        # Cubic term: real keystone from a rotated, tilted card is not
        # symmetric — one edge sinks while the other barely moves
        # (IMG_4948 cards 5/6 slicing number TOPS rightward). u^3 is odd,
        # so it needs no recentring.
        ccube = np.arange(-50, 50.1, 5)
        c0 = max(ccube, key=lambda v_: shear_score(shear, quad, v_))
        fcube = np.arange(c0 - 5, c0 + 5.01, 1)
        cube = float(max(fcube, key=lambda v_: shear_score(shear, quad, v_)))
        if abs(cube) < 4:
            cube = 0.0

    col_rows = []
    col_xcent = []
    micro = max(3, int(0.06 * ystep))
    for c in range(columns):
        xa = int(row_cuts[rows - 1][c])
        xb = int(row_cuts[rows - 1][c + 1])
        col_xcent.append(0.5 * (xa + xb))
        if xb - xa < 12:
            col_rows.append([int(v) for v in y_lines])
            continue
        prof = np.convolve(pm_full[:, xa:xb].mean(axis=1),
                           np.ones(7) / 7, mode="same")

        def run_centre_y(cand, a, b):
            t = prof[cand] + 0.015
            l_ = r_ = cand
            while l_ - 1 >= a and prof[l_ - 1] <= t:
                l_ -= 1
            while r_ + 1 < b and prof[r_ + 1] <= t:
                r_ += 1
            return (l_ + r_) // 2

        ys = []
        for k, yp in enumerate(y_lines):
            # Shear + keystone model first: boundary height at THIS
            # column's centre.
            u_c = (col_xcent[c] - xc) / hw
            yc = int(round(yp + shear * (col_xcent[c] - xc)
                           + quad * (u_c * u_c - 1.0 / 3.0)
                           + cube * (u_c * u_c * u_c)))
            yc = int(np.clip(yc, 0, h - 1))
            # Micro-snap only across a clean whitespace run — wide enough
            # to tidy the model, far too narrow to reach a digit gap.
            a, b = max(0, yc - micro), min(h, yc + micro + 1)
            ny = yc
            if b - a > 4:
                cand = run_centre_y(a + int(np.argmin(prof[a:b])), a, b)
                if prof[cand] <= 0.05:
                    ny = cand
            if ys and ny <= ys[-1]:
                ny = ys[-1] + max(3, int(0.10 * ystep))
            ys.append(min(ny, h - 1))
        col_rows.append(ys)

    # ----- Extract cells with modest inset to avoid grid-line ink ----------
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(columns):
            x1, x2 = row_cuts[r][c], row_cuts[r][c + 1]
            y1, y2 = col_rows[c][r], col_rows[c][r + 1]
            mx = max(3, (x2 - x1) // 10)
            my = max(3, (y2 - y1) // 10)
            cell = warped[y1 + my:y2 - my, x1 + mx:x2 - mx]
            row_cells.append(cell)
        cells.append(row_cells)

    # A pitch-based junk guard (rejecting ~60px columns as "torn
    # slivers") was tried and reverted: genuine cards span 47-113px per
    # column depending on how far the photo was taken from the sheet,
    # overlapping the sliver range completely. Junk fragments are left
    # to the caller to crop out of the source photos.
    if min_junk_pitch > 0 and rows_fitted and len(x_lines) > 1 and \
            float(np.median(np.diff(x_lines))) < min_junk_pitch:
        rows_fitted = False

    return cells, x_lines, y_lines, row_cuts, rows_fitted, col_rows




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
        strip = gray[y_top:y_bot]
        locmed = cv2.medianBlur(strip, 51).astype(np.int16)
        band = (strip.astype(np.int16) < locmed - 35).mean(axis=0)
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


def draw_grid_debug(card, x_lines, y_lines, filename, row_cuts=None,
                    col_rows=None):
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
    if col_rows is None:
        for y in y_lines:
            cv2.line(debug, (0, int(y)), (debug.shape[1] - 1, int(y)),
                     (0, 0, 255), 2)
    else:
        # Same idea for the horizontals: per-column refined boundaries,
        # drawn green so they are distinguishable from the vertical cuts.
        for c in range(len(col_rows)):
            if c + 1 >= len(row_cuts[0]):
                continue
            x1, x2 = row_cuts[len(row_cuts) - 1][c], \
                row_cuts[len(row_cuts) - 1][c + 1]
            for y in col_rows[c]:
                cv2.line(debug, (int(x1), int(y)), (int(x2), int(y)),
                         (0, 255, 0), 2)
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
        # Wide pieces are side-by-side cards even when internal row
        # gutters outnumber the column seams (IMG_4903): one card never
        # gets this wide relative to its height, so trust the vertical
        # seams first instead of letting row-gap count win the vote.
        if bw > 1.35 * bh and ccols:
            axis, cuts = 0, ccols
        else:
            axis = 0 if len(ccols) >= len(crows) else 1
            cuts = ccols if axis == 0 else crows
        if FIND_DEBUG:
            print(f"  [seg] recurse ({int(x)},{int(y)},{int(bw)}x{int(bh)}) "
                  f"d{depth} ccols={len(ccols)} crows={len(crows)} "
                  f"axis={'col' if axis == 0 else 'row'} ncuts={len(cuts)}")
        if not cuts:
            leaves.append([x, y, bw, bh, lid])
            return
        starts = [0] + [e + 1 for _, e in cuts]
        ends = [s for s, _ in cuts] + [(bw if axis == 0 else bh)]
        # A column seam flanked on both sides by over-wide pieces is a
        # false friend: real side-by-side cards are never each wider
        # than ~2.3x their height. IMG_4903's single wide cards carry a
        # mid-sheet shadow seam that used to be split on, destroying
        # the card. Keep such pieces whole.
        if axis == 0 and bw > 1.8 * bh:
            pieces = [(b - a) for a, b in zip(starts, ends)
                      if b - a >= 0.10 * w]
            if len(pieces) >= 2 and all(p > 2.3 * bh for p in pieces):
                leaves.append([x, y, bw, bh, lid])
                return
        for a, b in zip(starts, ends):
            if b - a < (0.10 * w if axis == 0 else 0.08 * h):
                continue
            if axis == 0:
                recurse(x + a, y, b - a, bh, lid, depth + 1)
            else:
                recurse(x, y + a, bw, b - a, lid, depth + 1)

    # Stacked cards hide from the projection splitter: printed sheets
    # put consecutive rows of cards nearly in contact, so the seam has
    # no deep print valley — while inside a single card the gaps
    # between number rows do. The BINGO headers rescue this: every card
    # starts with a saturated pink band, so a leaf holding several
    # well-separated pink bands is several vertically stacked cards,
    # with each band marking a card's top edge. Photos without colour
    # headers have no pink at all and pass through untouched.
    #
    # This runs on WHOLE blobs before recursion: recursing first can
    # carve a stacked sheet into cross-card slivers that no later stage
    # can reassemble (IMG_4903).

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
        # starting there. Once two bands are merged the local pitch is
        # known, so the lower bound tightens to just over half of it —
        # perspective can shrink far cards to ~0.6x the near ones
        # (IMG_4903), which the fixed 650px floor used to reject.
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
                lo = 650
                if len(merged) >= 2:
                    gaps = [abs(a[0] - b[0]) for ai, a in enumerate(merged)
                            for b in merged[ai + 1:]]
                    lo = min(650, int(0.55 * min(gaps)))
                if lo <= d <= 1500:
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

    # Two orderings, keep the better: splitting whole blobs on pink
    # card-tops BEFORE recursion rescues stacked-card sheets whose
    # texture blob spans everything (IMG_4903), but on other sheets the
    # root-scale pink peaks include FREE squares and title strips that
    # do not mark card tops — slicing there chops real cards into
    # aspect-dead strips (IMG_4910/4911). Recursion-first is the legacy
    # behaviour and stays the fallback. Both variants run to validated
    # leaf sets and whichever keeps more cards wins.
    def build_leaves(split_first):
        leaves.clear()
        bases = []
        for i in sorted(range(1, n_blobs),
                        key=lambda i_: -blob_stats[i_, cv2.CC_STAT_AREA]):
            x, y, bw, bh, area = blob_stats[i]
            if area < 0.005 * w * h:
                continue
            piece = [int(x), int(y), int(bw), int(bh), i]
            if split_first:
                bases.extend(split_pink(piece))
            else:
                bases.append(piece)
        for rx, ry, rw, rh, rlid in bases:
            recurse(rx, ry, rw, rh, rlid, 0)
        if not split_first:
            leaves[:] = [piece for l_ in leaves
                         for piece in split_pink(l_)]
        if FIND_DEBUG:
            tag = "split-first" if split_first else "recurse-first"
            print(f"  [seg] {tag} leaves: "
                  f"{[(int(a_), int(b_), int(c_), int(d_)) for a_, b_, c_, d_, _ in leaves]}")
        return [trim_leaf(l_) for l_ in leaves]

    def cheap_ok(lf):
        x, y, bw, bh, lid = lf
        if bw < 0.12 * w or bh < 0.08 * h:
            return False
        ar = bw / bh
        # Upper bound sits just above the widest real card observed —
        # IMG_4903's sheet carries landscape cards of aspect ~2.0 after
        # trim; junk strips are far beyond it, so a small margin costs
        # nothing.
        if not 0.55 <= ar <= 2.25:
            return False
        mm = blob_labels[y:y + bh, x:x + bw] == lid
        pm = np.where(mm, print_m[y:y + bh, x:x + bw], 0)
        return pm.sum() / 255 >= 0.05 * bw * bh

    def evidence_ok(lf):
        # Divider-evidence + quality gate: a leaf only counts as a card
        # if the FULL extraction pipeline accepts it (grid fit found AND
        # regularity/ink within limits). Mirror the crop geometry of
        # extract_card_grids (mask minAreaRect + quad growth + warp) —
        # a plain rect crop answers differently from what extraction
        # later decides (IMG_4944).
        try:
            x, y, bw, bh, lid = lf
            # Same padded window as extract_card_grids: the grown quad
            # can reach outside the leaf rect, and there it must sample
            # real photo pixels rather than black fill.
            px_, py_ = int(0.25 * bw), int(0.40 * bh)
            x0, y0 = max(0, x - px_), max(0, y - py_)
            x1 = min(image.shape[1], x + bw + px_)
            y1 = min(image.shape[0], y + bh + py_)
            sub = image[y0:y1, x0:x1]
            # Blob id regions can spill past a pink-cut piece's rect
            # (siblings share one component id); restrict to the leaf
            # rect exactly like the out_labels painting does, or the
            # minAreaRect swallows neighbouring cards.
            sm_full = (blob_labels[y0:y1, x0:x1] == lid)
            sm = np.zeros(sm_full.shape, np.uint8)
            yo, xo = y - y0, x - x0
            sm[yo:yo + bh, xo:xo + bw] = \
                sm_full[yo:yo + bh, xo:xo + bw]
            sm *= np.uint8(255)
            rect = cv2.minAreaRect(np.column_stack(
                np.nonzero(sm))[:, ::-1].astype(np.float32))
            box = order_points(cv2.boxPoints(rect))
            centre = box.mean(axis=0)
            box = centre + (box - centre) * (1.12, 1.28)
            ccard = four_point_warp(sub, box)
            sm_w = four_point_warp(sm, box)
            mys, mxs = np.nonzero(sm_w > 127)
            ty, by = int(mys.min()), int(mys.max())
            tx, bx = int(mxs.min()), int(mxs.max())
            pad_y = max(0, int(0.10 * (by - ty)))
            pad_x = max(0, int(0.06 * (bx - tx)))
            ty, tx = max(0, ty - pad_y), max(0, tx - pad_x)
            by = min(ccard.shape[0], by + pad_y)
            bx = min(ccard.shape[1], bx + pad_x)
            ccard = _deskew_header(deskew(
                ccard[ty:by, tx:bx]))
            cells_, xl_, yl_, rc_, fitted_, cr_ = \
                extract_grid_cells(ccard, min_junk_pitch=0)
            if not fitted_:
                if FIND_DEBUG:
                    print(f"  [seg] evidence ({x},{y},{bw}x{bh}) -> "
                          f"UNFITTED (rows/cols fit failed)")
                    cv2.imwrite(f"/tmp/gate_{x}_{y}.png", ccard)
                return False
            q = grid_quality(ccard, xl_, yl_, rc_)
            verdict = not (q["ink"] > 0.10 or q["y_reg"] > 0.30
                           or q["x_reg"] > 0.35)
            if FIND_DEBUG:
                print(f"  [seg] evidence ({x},{y},{bw}x{bh}) -> "
                      f"fitted={fitted_} q={q['y_reg']:.3f}/"
                      f"{q['x_reg']:.3f}/{q['ink']:.4f} ok={verdict}")
                cv2.imwrite(f"/tmp/gate_{x}_{y}.png", ccard)
            return verdict
        except Exception as exc:
            if FIND_DEBUG:
                print(f"  [seg] evidence ({lf[0]},{lf[1]}) EXC {exc}")
            return False

    variant_a = build_leaves(False)
    variant_b = build_leaves(True)

    # Cluster-union selection: neither leaf ordering wins everywhere.
    # Recurse-first strips can truncate a card edge (IMG_4903 card 1),
    # while split-first pink cuts can shear off header-only slabs that
    # are not full cards (IMG_4910). Leaves from both variants that
    # cover the same card region form one cluster; the cluster keeps
    # its best gate-passing member, preferring split-first (pink-
    # anchored boundaries). A cluster with no passing member is dropped.
    cand_b = [lf for lf in sorted(variant_b, key=lambda l_: l_[1])
              if cheap_ok(lf)]
    cand_a = [lf for lf in sorted(variant_a, key=lambda l_: l_[1])
              if cheap_ok(lf)]

    def frac_small(r1, r2):
        ix = max(0, min(r1[0] + r1[2], r2[0] + r2[2]) -
                 max(r1[0], r2[0]))
        iy = max(0, min(r1[1] + r1[3], r2[1] + r2[3]) -
                 max(r1[1], r2[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        return inter / min(r1[2] * r1[3], r2[2] * r2[3])

    clusters = []          # each: {"members":[(lf,variant)]}
    for tag_, cands in (("B", cand_b), ("A", cand_a)):
        for lf in cands:
            best = None
            for cl in clusters:
                # Match against member rects only: growing the cluster
                # rectangle lets a wide strip bridge two neighbouring
                # cards into one cluster and swallow one of them
                # (IMG_4955).
                if any(frac_small(m[0][:4], lf[:4]) >= 0.55
                       for m in cl["members"]):
                    best = cl
                    break
            if best is None:
                clusters.append({"members": [(lf, tag_)]})
            else:
                best["members"].append((lf, tag_))

    out_labels = np.zeros(image.shape[:2], np.int32)
    groups = []
    nid = 1
    n_dropped = 0
    for cl in sorted(clusters,
                     key=lambda c_: min(m[0][1] for m in c_["members"])):
        members = sorted(cl["members"], key=lambda m_: m_[1])  # B first
        chosen = None
        for lf, _tag in members:
            if evidence_ok(lf):
                chosen = lf
                break
        if chosen is None:
            n_dropped += 1
            if FIND_DEBUG:
                print(f"  [seg] cluster "
                      f"@{[m[0][:4] for m in cl['members']]} dropped: "
                      f"no divider evidence")
            continue
        x, y, bw, bh, lid = chosen
        mm = blob_labels[y:y + bh, x:x + bw] == lid
        out_labels[y:y + bh, x:x + bw][mm] = nid
        groups.append({"ids": [nid],
                       "bbox": (int(x), int(y), int(bw), int(bh))})
        nid += 1
    if FIND_DEBUG:
        print(f"  [seg] clusters={len(clusters)} kept={len(groups)} "
              f"(A leaves={len(cand_a)}, B leaves={len(cand_b)})")
    if groups:
        med_area = float(np.median(
            [g["bbox"][2] * g["bbox"][3] for g in groups]))
        keep = [g for g in groups
                if g["bbox"][2] * g["bbox"][3] >= 0.25 * med_area]
        if len(keep) != len(groups):
            bad = {g["ids"][0] for g in groups} \
                - {g["ids"][0] for g in keep}
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
            # Photos of bare tables (IMG_4945) legitimately contain no
            # cards; report and move on instead of killing a whole
            # batch run.
            print("Could not detect the outer sheet boundary — "
                  "no cards found")
            return card_images

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
        # smears and slices through digits. The text-based estimate
        # runs first; the header strip gives a second, colour-anchored
        # read for cards that sit tilted inside an axis-aligned leaf,
        # which plain deskew cannot see. Done before saving so every
        # debug image and cell crop shares the same coordinates.
        angle = estimate_skew_angle(card)
        card = deskew(card)
        rot = _deskew_header(card)
        if rot is not card:
            angle2 = estimate_skew_angle(rot)
            print(f"  {card_name}: header-levelled by "
                  f"{angle - angle2:+.2f} deg")
            card = rot
        print(f"  {card_name}: deskewed by {angle:+.2f} deg")

        cv2.imwrite(str(output_dir / f"{card_name}.png"), card)

        cells, x_lines, y_lines, row_cuts, rows_fitted, col_rows = \
            extract_grid_cells(card)
        if not rows_fitted:
            print(f"  {card_name}: REJECTED (no divider evidence — not a card)")
            cv2.imwrite(str(output_dir / f"{card_name}_rejected.png"), card)
            continue
        q = grid_quality(card, x_lines, y_lines, row_cuts)
        # Tiny AND dense = background fragment, not a card. Either
        # signal alone is inconclusive — genuine cards photographed up
        # close run 47-113px per column, and heavily daubed cards carry
        # real ink — but a grid barely 60px wide whose cells hold >5%
        # ink is scribble on a torn sliver (IMG_4948 card 1: ink 0.083;
        # the smallest genuine card in the corpus sits at ink 0.002).
        if q["ink"] > 0.05 and len(x_lines) > 1 and \
                float(np.median(np.diff(x_lines))) < 70:
            print(f"  {card_name}: REJECTED (tiny dense fragment "
                  f"— ink={q['ink']:.3f} at "
                  f"{float(np.median(np.diff(x_lines))):.0f}px columns)")
            cv2.imwrite(str(output_dir / f"{card_name}_rejected.png"), card)
            continue
        print(f"  {card_name}: quality y_reg={q['y_reg']:.3f} "
              f"x_reg={q['x_reg']:.3f} ink={q['ink']:.4f}")
        if q["ink"] > 0.10 or q["y_reg"] > 0.30 or q["x_reg"] > 0.35:
            print(f"  {card_name}: REJECTED (grid does not fit a 5x5 card)")
            cv2.imwrite(str(output_dir / f"{card_name}_rejected.png"), card)
            continue
        save_cells(cells, output_dir / "cells", card_name)
        draw_grid_debug(card, x_lines, y_lines,
                        output_dir / f"{card_name}_grid_debug.png",
                        row_cuts=row_cuts, col_rows=col_rows)

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
