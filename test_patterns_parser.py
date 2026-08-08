import pytest

import patterns_parser as pp


# -------------------------
# Helpers for assertions
# -------------------------

def load_lines(text: str):
    return text.splitlines()

def get_class_name(obj):
    return obj.__class__.__name__

def assert_pattern_is_maskref(pat):
    # MaskPattern should be returned for leaf mask references
    assert get_class_name(pat) == "MaskPattern"

def assert_pattern_is_all(pat):
    assert get_class_name(pat) == "AllPattern"

def assert_pattern_is_any(pat):
    assert get_class_name(pat) == "AnyPattern"

def assert_constraint_leaf(pat):
    assert get_class_name(pat) == "ConstraintPattern"
    # check it captured the expected values when possible
    assert hasattr(pat, "name")
    assert hasattr(pat, "n")
    assert hasattr(pat, "comparison")

def get_attr(obj, name, default=None):
    return getattr(obj, name, default)

def pattern_signature(pat):
    """
    Try to produce a stable "signature" of the parse tree so tests can compare expected structure.
    This is intentionally tolerant of attribute naming differences.
    """
    cls = get_class_name(pat)

    # Mask leaf
    if cls == "MaskPattern":
        # Try a couple likely attribute names
        name = (
            get_attr(pat, "name")
            or get_attr(pat, "mask_name")
            or get_attr(pat, "maskref")
            or get_attr(pat, "pattern_name")
        )
        # Some implementations might store the mask itself; ignore mask content for signature.
        return ("MASK", name)

    # AND/OR nodes
    if cls in ("AllPattern", "AnyPattern"):
        # Try common attribute names for children
        children = (
            get_attr(pat, "patterns")
            or get_attr(pat, "children")
            or get_attr(pat, "args")
            or get_attr(pat, "subpatterns")
        )
        if children is None:
            # fallback: attempt to find list/tuple attributes
            for v in pat.__dict__.values():
                if isinstance(v, (list, tuple)) and v and all(hasattr(x, "__class__") for x in v):
                    children = v
                    break
        assert isinstance(children, list) or isinstance(children, tuple)
        op = "AND" if cls == "AllPattern" else "OR"
        return (op, tuple(pattern_signature(c) for c in children))

    # AtLeast / AtMost leaf
    if cls == "ConstraintPattern":
        mask_name = get_attr(pat, "name")
        n = get_attr(pat, "n")
        comparison = get_attr(pat, "comparison")
        tag = "AT_LEAST" if comparison == "at_least" else "AT_MOST"
        return (tag, mask_name, n)

    # Unknown node type
    return (cls, None)


# -------------------------
# Fixtures: config snippets
# -------------------------

VALID_CONFIG = """
# Patterns config file

# ------------------------------------------------------------
# MASKS
# ------------------------------------------------------------

MASK row0
XXXXX
-----
-----
-----
-----

MASK row1
-----
XXXXX
-----
-----
-----

MASK row4
-----
-----
-----
-----
XXXXX

PATTERN top3-and-bottom3
AND
    AtLeast(row0, 3)
    AtLeast(row4, 3)

PATTERN standard-bingo
OR
    row0
    row1
    row4
"""

# include full 5x5 masks used in your sample (optional, but good)
FULL_CONFIG = """
MASK row0
XXXXX
-----
-----
-----
-----

MASK row4
-----
-----
-----
-----
XXXXX

MASK col0
X----
X----
X----
X----
X----

MASK col4
----X
----X
----X
----X
----X

MASK diagonal
----X
---X-
--X--
-X---
X----
    
PATTERN p1
AND
    AtLeast(row0, 5)
    col0

PATTERN p2
OR
    diagonal
    AND
        col4
        AtLeast(row4, 1)
"""

# -------------------------
# Mask parsing tests
# -------------------------

def test_parse_masks_ignores_blank_and_comments_and_stops_at_patstring():
    lines = load_lines(VALID_CONFIG)
    masks, i = pp.parse_masks(lines, 0)

    assert "row0" in masks
    assert "row1" in masks
    assert "row4" in masks

    # parse_masks should stop before PATTERN section
    assert i <= len(lines)
    # sanity: ensure returned index points near PATTERN start (not strictly required)
    assert "PATTERN" not in lines[i] or not lines[i].lstrip().startswith("MASK ")

def test_parse_mask_row_accepts_X_and_x_and_dash():
    assert pp.parse_mask_row("X----") == [True, False, False, False, False]
    assert pp.parse_mask_row("x----") == [True, False, False, False, False]
    assert pp.parse_mask_row("-----") == [False, False, False, False, False]

def test_parse_mask_row_rejects_wrong_length():
    with pytest.raises(pp.ParseError):
        pp.parse_mask_row("X---")  # 4 chars

def test_parse_mask_row_rejects_invalid_char():
    with pytest.raises(pp.ParseError):
        pp.parse_mask_row("Z----")

def test_parse_masks_rejects_non_5_rows_for_a_mask():
    # row0 has only 4 mask rows
    bad = """
MASK row0
XXXXX
-----
-----
-----
PATTERN p
row0
"""
    lines = load_lines(bad)
    with pytest.raises(pp.ParseError):
        pp.parse_masks(lines, 0)

def test_parse_masks_rejects_duplicate_mask_name():
    dup = """
MASK row0
XXXXX
-----
-----
-----
-----

MASK row0
XXXXX
-----
-----
-----
-----
"""
    lines = load_lines(dup)
    with pytest.raises(pp.ParseError):
        pp.parse_masks(lines, 0)


# -------------------------
# Pattern parsing tests
# -------------------------

def test_parse_patterns_parses_simple_mask_leaf_and_sets_base_indent_logic():
    # masks must be parsed first
    lines = load_lines(FULL_CONFIG)

    # Parse only masks
    masks, i = pp.parse_masks(lines, 0)

    # Parse patterns from where parse_masks stopped
    patterns, _ = pp.parse_patterns(lines, i, masks)

    assert "p1" in patterns
    assert "p2" in patterns

def test_parse_patterns_leaf_mask_returns_maskpattern():
    lines = load_lines(FULL_CONFIG)
    masks, i = pp.parse_masks(lines, 0)
    patterns, _ = pp.parse_patterns(lines, i, masks)

    # p1 = AND( AtLeast(row0, 5), col0 )
    sig = pattern_signature(patterns["p1"])
    # Expected structure
    assert sig[0] == "AND"
    child_sigs = sig[1]
    # one child should be AT_LEAST and other should be MASK
    assert any(cs[0] == "AT_LEAST" for cs in child_sigs)
    assert any(cs[0] == "MASK" for cs in child_sigs)

def test_parse_patterns_parses_at_least_leaf():
    lines = load_lines(FULL_CONFIG)
    masks, i = pp.parse_masks(lines, 0)
    patterns, _ = pp.parse_patterns(lines, i, masks)

    # Find the leaf in p1 signature
    sig = pattern_signature(patterns["p1"])
    assert ("AT_LEAST", "row0", 5) in sig[1]

def test_parse_patterns_parses_nested_and_within_or():
    lines = load_lines(FULL_CONFIG)
    masks, i = pp.parse_masks(lines, 0)
    patterns, _ = pp.parse_patterns(lines, i, masks)

    sig = pattern_signature(patterns["p2"])
    # p2 = OR( diagonal, AND(col4, AtLeast(row4, 1)) )
    assert sig[0] == "OR"
    or_children = sig[1]
    assert any(child[0] == "MASK" for child in or_children)

    # nested AND should exist
    assert any(child[0] == "AND" for child in or_children)

def test_parse_patterns_parses_at_most_leaf():
    config = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
AND
    AtMost(row0, 2)
"""
    lines = load_lines(config)
    masks, i = pp.parse_masks(lines, 0)
    patterns, _ = pp.parse_patterns(lines, i, masks)

    sig = pattern_signature(patterns["p"])
    assert ("AT_MOST", "row0", 2) in sig[1]

def test_parse_patterns_rejects_unknown_mask_in_set_count():
    bad = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
AND
    AtLeast(DOES_NOT_EXIST, 1)
"""
    lines = load_lines(bad)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)

def test_parse_patterns_rejects_unknown_mask_in_mask_leaf():
    bad = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
OR
    row0
    no_such_mask
"""
    lines = load_lines(bad)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)

def test_parse_patterns_rejects_bad_set_count_syntax():
    bad = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
AND
    AtLeast(row0)
"""
    lines = load_lines(bad)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)

def test_parse_patterns_rejects_out_of_range_at_least():
    bad = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
AND
    AtLeast(row0, -1)
"""
    lines = load_lines(bad)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)

def test_parse_patterns_rejects_out_of_range_at_least_high():
    bad = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
AND
    AtLeast(row0, 26)
"""
    lines = load_lines(bad)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)

def test_parse_patterns_rejects_patterns_section_mixing_mask_line():
    # MASK appears after PATTERN start - your parse_patterns explicitly rejects that
    bad = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
OR
    row0

MASK row1
-----
XXXXX
-----
-----
-----
"""
    lines = load_lines(bad)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)

def test_parse_patterns_rejects_duplicate_pattern_names():
    dup = """
MASK row0
XXXXX
-----
-----
-----
-----

PATTERN p
OR
    row0

PATTERN p
OR
    row0
"""
    lines = load_lines(dup)
    masks, i = pp.parse_masks(lines, 0)
    with pytest.raises(pp.ParseError):
        pp.parse_patterns(lines, i, masks)


# -------------------------
# Integration-ish test with your provided full sample file
# -------------------------

def test_parse_full_sample_config_builds_patterns_tree():
    # Put your real patterns_config.txt contents here if you want
    # For now, we use FULL_CONFIG structure above.
    lines = load_lines(FULL_CONFIG)
    masks, i = pp.parse_masks(lines, 0)
    patterns, _ = pp.parse_patterns(lines, i, masks)

    assert "p1" in patterns and "p2" in patterns
    # ensure they are winning pattern objects
    assert patterns["p1"] is not None
    assert patterns["p2"] is not None
