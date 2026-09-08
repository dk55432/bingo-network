"""
bingo_header_locator.py

Detects one or more "BINGO" header instances in a photographed/scanned sheet
by matching against clean flatbed-scanned template samples (one per color
variant), then projects precisely-measured column-boundary points (B|I, I|N,
N|G, G|O) from template space into scene space using the recovered
homography for each instance.

Because the homography captures rotation, scale, and perspective (keystone)
together, any point you define once on the clean template -- not just
detected keypoints -- can be carried into the photo correctly transformed.

Usage:
    python bingo_header_locator.py scene.jpg template_orange.jpg template_blue.jpg ...

Dependencies:
    pip install opencv-python numpy
"""

import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Define your column boundaries ONCE, as fractions of template width.
#    Measure these precisely on a clean scan (e.g. in an image editor,
#    read off pixel x / total width). Y fractions define where along the
#    header's vertical extent the point sits (0.0 = top edge, 1.0 = bottom
#    edge of the header strip you scanned).
# ---------------------------------------------------------------------------
COLUMN_BOUNDARIES_X_FRAC = {
    "B|I": 0.19,
    "I|N": 0.385,   # placeholders -- replace with your measured values
    "N|G": 0.57,
    "G|O": 0.75,
}
TOP_Y_FRAC = 0.0
BOTTOM_Y_FRAC = 1.0

# ---------------------------------------------------------------------------
# 1b. Row boundaries, expressed as fractions of the FULL CARD height (not
#     just the header crop). Origin (0.0) is the same y=0 as the header
#     template's top edge. Measure these once from a full-card reference
#     scan at the same pixel scale as your header templates.
# ---------------------------------------------------------------------------
ROW_BOUNDARIES_Y_FRAC = {
    "top_edge":       0.0,     # placeholders -- replace with your measured values
    "above_row1":     0.13,
    "row1|row2":      0.25,
    "row2|row3":      0.40,
    "row3|row4":      0.57,  # example -- measure your actual value
    "row4|row5":      0.74,  # example -- measure your actual value
    "bottom_edge":    1.0,
}

# How many header-crop-heights tall the full card is, measured once from a
# reference scan at the same pixel scale as your header templates:
#   CARD_TO_HEADER_RATIO = full_card_height_px / header_crop_height_px
##CARD_TO_HEADER_RATIO = (1132/278*2)  # placeholder -- replace with your measured value
CARD_TO_HEADER_RATIO = 8

# ---------------------------------------------------------------------------
# 2. Matching / detection tuning
# ---------------------------------------------------------------------------
RATIO_TEST_THRESHOLD = 0.75      # Lowe's ratio test
MIN_GOOD_MATCHES = 12            # below this, stop looking for more instances
RANSAC_REPROJ_THRESHOLD = 5.0
MAX_INSTANCES = 3                # you said sheets have 1-3 cards


def load_template_features(detector, template_paths):
    """Extract keypoints/descriptors for every color-variant template."""
    templates = []
    for path in template_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read template: {path}")
        kp, des = detector.detectAndCompute(img, None)
        h, w = img.shape[:2]
        templates.append({
            "path": path,
            "img": img,
            "kp": kp,
            "des": des,
            "w": w,
            "h": h,
        })
    return templates


def template_boundary_points(template):
    """Return the column boundary points (top & bottom) in template pixel space."""
    w, h = template["w"], template["h"]
    pts = {}
    for name, xf in COLUMN_BOUNDARIES_X_FRAC.items():
        top = (xf * w, TOP_Y_FRAC * h)
        bot = (xf * w, BOTTOM_Y_FRAC * h)
        pts[name] = (top, bot)
    return pts


def template_row_points(template):
    """
    Return the row boundary points (left & right) in template pixel space.
    y is expressed as a fraction of the FULL CARD height, so it legitimately
    extends past the header crop's own pixel height -- that's expected and
    fine, since the homography describes the whole flat card, not just the
    cropped header region.
    """
    w, h = template["w"], template["h"]
    card_height_px = h * CARD_TO_HEADER_RATIO
    pts = {}
    for name, yf in ROW_BOUNDARIES_Y_FRAC.items():
        y = yf * card_height_px
        left = (0.0, y)
        right = (w, y)
        pts[name] = (left, right)
    return pts


def find_best_instance(matcher, template, scene_kp, scene_des, active_mask):
    """
    Match template descriptors against the currently-active subset of scene
    keypoints/descriptors, run RANSAC, and return the homography + inlier
    scene keypoint indices if a confident match is found, else None.
    """
    active_indices = np.where(active_mask)[0]
    if len(active_indices) < MIN_GOOD_MATCHES:
        return None

    sub_des = scene_des[active_indices]
    matches = matcher.knnMatch(template["des"], sub_des, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([template["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # map back from sub-index to original scene keypoint index
    dst_pts = np.float32(
        [scene_kp[active_indices[m.trainIdx]].pt for m in good]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() < MIN_GOOD_MATCHES:
        return None

    inlier_scene_indices = active_indices[[m.trainIdx for i, m in enumerate(good) if inlier_mask[i]]]

    return {
        "H": H,
        "num_inliers": int(inlier_mask.sum()),
        "inlier_scene_indices": inlier_scene_indices,
    }


def detect_all_instances(detector, matcher, templates, scene_img):
    """
    Iteratively detect up to MAX_INSTANCES header instances in the scene,
    trying every color-variant template each round and keeping the best
    match. Matched scene keypoints are removed before the next round so the
    same physical header isn't found twice.
    """
    scene_kp, scene_des = detector.detectAndCompute(scene_img, None)
    if scene_des is None or len(scene_kp) == 0:
        return []

    active_mask = np.ones(len(scene_kp), dtype=bool)
    instances = []

    for _ in range(MAX_INSTANCES):
        best = None
        best_template = None
        for template in templates:
            result = find_best_instance(matcher, template, scene_kp, scene_des, active_mask)
            if result and (best is None or result["num_inliers"] > best["num_inliers"]):
                best = result
                best_template = template

        if best is None:
            break  # no more confident matches left

        instances.append({"H": best["H"], "template": best_template})
        active_mask[best["inlier_scene_indices"]] = False  # consume these keypoints

    return instances


def project_boundaries(instance):
    """
    Project the template's column-boundary points into scene coordinates
    for one detected header instance.
    """
    template = instance["template"]
    H = instance["H"]
    boundary_pts_template = template_boundary_points(template)

    projected = {}
    for name, (top, bot) in boundary_pts_template.items():
        pts = np.float32([top, bot]).reshape(-1, 1, 2)
        scene_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        projected[name] = {
            "top": tuple(scene_pts[0]),
            "bottom": tuple(scene_pts[1]),
        }
    return projected


def project_rows(instance):
    """
    Project this instance's row-boundary points into scene coordinates.
    Unlike columns, rows don't need connecting across multiple header
    instances -- one card's own H already covers its entire grid, since the
    homography describes the whole flat card plane, not just the header crop.
    """
    template = instance["template"]
    H = instance["H"]
    row_pts_template = template_row_points(template)

    projected = {}
    for name, (left, right) in row_pts_template.items():
        pts = np.float32([left, right]).reshape(-1, 1, 2)
        scene_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        projected[name] = {
            "left": tuple(scene_pts[0]),
            "right": tuple(scene_pts[1]),
        }
    return projected


def sort_instances_top_to_bottom(instances, projections):
    """Order detected headers by their vertical position on the sheet."""
    def y_key(item):
        _, proj = item
        return proj["B|I"]["top"][1]
    return sorted(zip(instances, projections), key=y_key)


def build_full_column_lines(ordered_projections, scene_height):
    """
    For each column boundary, connect the corresponding point across
    consecutive header instances to build a full line spanning the sheet.
    Extrapolates above the first header and below the last header using the
    slope from the nearest pair of points.
    """
    boundary_names = COLUMN_BOUNDARIES_X_FRAC.keys()
    lines = {}

    for name in boundary_names:
        top_points = [proj[name]["top"] for _, proj in ordered_projections]
        bottom_points = [proj[name]["bottom"] for _, proj in ordered_projections]

        # interleave top/bottom per card to get one polyline down the sheet
        polyline = []
        for t, b in zip(top_points, bottom_points):
            polyline.append(t)
            polyline.append(b)

        lines[name] = {
            "polyline": polyline,  # draw as connected segments down the sheet
        }

        # Optional: extrapolate a bit above the first / below the last point
        # using the slope between the nearest two points, in case you want
        # the line to run to the physical sheet edges.
        if len(polyline) >= 2:
            (x0, y0), (x1, y1) = polyline[0], polyline[1]
            slope = (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else 0
            top_extrap_y = 0
            top_extrap_x = x0 - slope * (y0 - top_extrap_y)
            lines[name]["extrapolated_top"] = (top_extrap_x, top_extrap_y)

            (x0, y0), (x1, y1) = polyline[-2], polyline[-1]
            slope = (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else 0
            bot_extrap_y = scene_height
            bot_extrap_x = x1 + slope * (bot_extrap_y - y1)
            lines[name]["extrapolated_bottom"] = (bot_extrap_x, bot_extrap_y)

    return lines


def draw_debug_overlay(scene_color, lines, row_projections=None):
    """Draw the computed column and row lines on a copy of the scene for visual QA."""
    overlay = scene_color.copy()
    for name, data in lines.items():
        pts = data["polyline"]
        if "extrapolated_top" in data:
            pts = [data["extrapolated_top"]] + pts
        if "extrapolated_bottom" in data:
            pts = pts + [data["extrapolated_bottom"]]
        pts_int = [(int(round(x)), int(round(y))) for x, y in pts]
        for p0, p1 in zip(pts_int, pts_int[1:]):
            cv2.line(overlay, p0, p1, (0, 0, 255), 2)  # columns in red

    if row_projections:
        for proj in row_projections:
            for name, pts in proj.items():
                left = tuple(int(round(v)) for v in pts["left"])
                right = tuple(int(round(v)) for v in pts["right"])
                cv2.line(overlay, left, right, (255, 0, 0), 2)  # rows in blue

    return overlay


def main():
    if len(sys.argv) < 3:
        print("Usage: python bingo_header_locator.py scene.jpg template1.jpg [template2.jpg ...]")
        sys.exit(1)

    scene_path = sys.argv[1]
    template_paths = sys.argv[2:]

    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    templates = load_template_features(detector, template_paths)

    scene_color = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)

    instances = detect_all_instances(detector, matcher, templates, scene_gray)
    print(f"Detected {len(instances)} header instance(s).")

    if not instances:
        return

    projections = [project_boundaries(inst) for inst in instances]
    ordered = sort_instances_top_to_bottom(instances, projections)

    row_projections = []
    for i, (inst, proj) in enumerate(ordered):
        print(f"\nInstance {i} (template: {inst['template']['path']}):")
        for name, pts in proj.items():
            print(f"  {name}: top={pts['top']}, bottom={pts['bottom']}")

        row_proj = project_rows(inst)
        row_projections.append(row_proj)
        print(f"  -- rows (own homography, no cross-card connection needed) --")
        for name, pts in row_proj.items():
            print(f"  {name}: left={pts['left']}, right={pts['right']}")

    lines = build_full_column_lines(ordered, scene_color.shape[0])
    overlay = draw_debug_overlay(scene_color, lines, row_projections)
    out_path = "bingo_columns_debug.png"
    cv2.imwrite(out_path, overlay)
    print(f"\nDebug overlay written to {out_path}")


if __name__ == "__main__":
    main()
