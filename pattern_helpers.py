"""
Shared helpers for building masks and WinningPattern trees.

Deliberately depends on nothing but winning_pattern.py — no Game, no
Player, no BingoCard, no fastapi. Both test_winning_pattern_unit.py and
test_winning_pattern_integration.py import from here, so the mask/pattern
construction logic lives in exactly one place, without either test file
picking up dependencies it doesn't actually need.

Not a test file itself (no test_ prefix), so pytest won't try to collect
it directly.
"""

from winning_pattern import MaskPattern, AnyPattern, AllPattern


def blank_mask():
    return [[False] * 5 for _ in range(5)]


def row_mask(row):
    mask = blank_mask()
    for col in range(5):
        mask[row][col] = True
    return mask


def col_mask(col):
    mask = blank_mask()
    for row in range(5):
        mask[row][col] = True
    return mask


def diagonal_mask(direction):
    """direction: '\\' for top-left to bottom-right, '/' for the other way."""
    mask = blank_mask()
    for i in range(5):
        col = i if direction == "\\" else 4 - i
        mask[i][col] = True
    return mask


def combine_masks(*masks):
    """OR multiple masks together. Useful for building a `marked` grid that
    satisfies several requirements at once — e.g. combine_masks(row_mask(0),
    diagonal_mask("\\")) gives you marks covering both row 0 and the
    diagonal, for testing nested AnyPattern/AllPattern combinations."""
    combined = blank_mask()
    for mask in masks:
        for row in range(5):
            for col in range(5):
                if mask[row][col]:
                    combined[row][col] = True
    return combined


def standard_bingo_pattern():
    """Any completed row, column, or diagonal — the usual bingo win."""
    masks = [row_mask(r) for r in range(5)] + [col_mask(c) for c in range(5)]
    masks += [diagonal_mask("\\"), diagonal_mask("/")]
    mask_patterns = [MaskPattern(f"pattern-{i}", m) for i, m in enumerate(masks)]
    return AnyPattern(mask_patterns)


def both_diagonals_pattern():
    """Both diagonals completed at once (forms a big X) — needs AllPattern."""
    masks = [diagonal_mask("\\"), diagonal_mask("/")]
    mask_patterns = [MaskPattern(f"pattern-{i}", m) for i, m in enumerate(masks)]
    return AllPattern(mask_patterns)
