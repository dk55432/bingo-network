"""
Unit tests for winning_pattern.py — MaskPattern, AnyPattern, AllPattern
tested in complete isolation.

No Game, no Player, no real BingoCard, no fastapi anywhere in this file's
import chain (check pattern_helpers.py — same story there). That's
deliberate: these tests should always be runnable, fast, and immune to
anything breaking elsewhere in the project. If these fail, the bug is
in winning_pattern.py itself, full stop — nothing else to suspect.

For tests that exercise the real Game/Player/card-marking pipeline, see
test_winning_pattern_integration.py instead.

Run with: pytest test_winning_pattern_unit.py -v
"""

from types import SimpleNamespace

import pytest

from winning_pattern import MaskPattern, AnyPattern, AllPattern, ConstraintPattern
from pattern_helpers import blank_mask, row_mask, diagonal_mask, combine_masks


# ---------------------------------------------------------------------------
# MaskPattern
#
# MaskPattern.matches() only ever reads card.marked, so a SimpleNamespace
# with just that attribute stands in for a card — no need to construct a
# real BingoCard (grid, card_id, etc.) just to test this one class.
# ---------------------------------------------------------------------------

class TestMaskPattern:
    def _card_with_marks(self, marked):
        return SimpleNamespace(marked=marked)

    def test_matches_when_every_masked_cell_is_marked(self):
        marked = row_mask(2)  # happens to double as a "fully marked row 2" grid
        pattern = MaskPattern("row-2", row_mask(2))
        assert pattern.matches(self._card_with_marks(marked))

    def test_does_not_match_when_one_masked_cell_is_unmarked(self):
        marked = row_mask(2)
        marked[2][3] = False  # one cell in the required row isn't marked
        pattern = MaskPattern("row-2", row_mask(2))
        assert not pattern.matches(self._card_with_marks(marked))

    def test_unmasked_cells_are_irrelevant(self):
        # Only row 0 is marked; mask only requires row 0. Should match
        # regardless of what the rest of the card looks like.
        marked = row_mask(0)
        pattern = MaskPattern("row-0", row_mask(0))
        assert pattern.matches(self._card_with_marks(marked))

    def test_empty_mask_always_matches(self):
        # An all-False mask requires nothing, so any card satisfies it —
        # an edge case worth pinning down explicitly.
        pattern = MaskPattern("empty", blank_mask())
        assert pattern.matches(self._card_with_marks(blank_mask()))


# ---------------------------------------------------------------------------
# ConstraintPattern
#
# Same SimpleNamespace-as-card approach as MaskPattern above — matches()
# only reads card.marked, so no need for a real BingoCard here either.
# ---------------------------------------------------------------------------

class TestConstraintPattern:
    def _card_with_marks(self, marked):
        return SimpleNamespace(marked=marked)

    def _marked_count(self, n):
        # A row-0 mask with exactly n of its 5 cells marked — enough to
        # exercise counting without needing a full row completed.
        marked = blank_mask()
        for col in range(n):
            marked[0][col] = True
        return marked

    def test_at_least_matches_when_count_exceeds_threshold(self):
        pattern = ConstraintPattern.at_least("row-0", row_mask(0), n=3)
        assert pattern.matches(self._card_with_marks(self._marked_count(4)))

    def test_at_least_matches_at_exact_threshold(self):
        pattern = ConstraintPattern.at_least("row-0", row_mask(0), n=3)
        assert pattern.matches(self._card_with_marks(self._marked_count(3)))

    def test_at_least_does_not_match_below_threshold(self):
        pattern = ConstraintPattern.at_least("row-0", row_mask(0), n=3)
        assert not pattern.matches(self._card_with_marks(self._marked_count(2)))

    def test_at_most_matches_at_exact_threshold(self):
        pattern = ConstraintPattern.at_most("row-0", row_mask(0), n=3)
        assert pattern.matches(self._card_with_marks(self._marked_count(3)))

    def test_at_most_matches_below_threshold(self):
        pattern = ConstraintPattern.at_most("row-0", row_mask(0), n=3)
        assert pattern.matches(self._card_with_marks(self._marked_count(2)))

    def test_at_most_does_not_match_above_threshold(self):
        pattern = ConstraintPattern.at_most("row-0", row_mask(0), n=3)
        assert not pattern.matches(self._card_with_marks(self._marked_count(4)))

    def test_only_counts_cells_inside_the_mask(self):
        # 3 cells marked, but only 1 of them is actually inside the mask
        # (row 0) — should count as 1, not 3.
        marked = blank_mask()
        marked[0][0] = True   # inside the row-0 mask
        marked[3][0] = True   # outside it
        marked[4][4] = True   # outside it
        pattern = ConstraintPattern.at_least("row-0", row_mask(0), n=2)
        assert not pattern.matches(self._card_with_marks(marked))

    def test_rejects_unknown_comparison(self):
        with pytest.raises(ValueError):
            ConstraintPattern("row-0", row_mask(0), "at_most_or_something", 3)


# ---------------------------------------------------------------------------
# AnyPattern / AllPattern
#
# These classes only ever call pattern.matches(card) on their children, so
# a stub that returns a fixed bool is enough to test the any/all combining
# logic on its own — no need for real MaskPatterns here either.
# ---------------------------------------------------------------------------

class _StubPattern:
    def __init__(self, result):
        self.result = result

    def matches(self, card):
        return self.result


class TestAnyPattern:
    def test_true_if_any_pattern_matches(self):
        patterns = [_StubPattern(False), _StubPattern(True), _StubPattern(False)]
        assert AnyPattern(patterns).matches(card=None)

    def test_false_if_no_pattern_matches(self):
        patterns = [_StubPattern(False), _StubPattern(False)]
        assert not AnyPattern(patterns).matches(card=None)

    def test_false_for_empty_pattern_list(self):
        assert not AnyPattern([]).matches(card=None)


class TestAllPattern:
    def test_true_if_every_pattern_matches(self):
        patterns = [_StubPattern(True), _StubPattern(True)]
        assert AllPattern(patterns).matches(card=None)

    def test_false_if_any_pattern_fails(self):
        patterns = [_StubPattern(True), _StubPattern(False)]
        assert not AllPattern(patterns).matches(card=None)

    def test_true_for_empty_pattern_list(self):
        # Vacuously true ("every pattern in an empty list matches") —
        # worth asserting explicitly so this behavior is a documented
        # choice, not just an accident nobody noticed.
        assert AllPattern([]).matches(card=None)


# ---------------------------------------------------------------------------
# Nested patterns — AnyPattern and AllPattern composed of each other, not
# just of plain MaskPatterns. This is what lets you express something like
# "(row 0 or row 4) and diagonal \\" as a single WinningPattern tree, the
# same way the real pattern-definition code will build these.
# ---------------------------------------------------------------------------

class TestNestedPatterns:
    """(row 0 OR row 4) AND diagonal \\ """

    def _card_with_marks(self, marked):
        return SimpleNamespace(marked=marked)

    def _build_pattern(self):
        either_row = AnyPattern([
            MaskPattern("row-0", row_mask(0)),
            MaskPattern("row-4", row_mask(4)),
        ])
        diagonal = MaskPattern("diagonal-\\", diagonal_mask("\\"))
        return AllPattern([either_row, diagonal])

    def test_matches_row0_and_diagonal(self):
        pattern = self._build_pattern()
        marked = combine_masks(row_mask(0), diagonal_mask("\\"))
        assert pattern.matches(self._card_with_marks(marked))

    def test_matches_row4_and_diagonal(self):
        pattern = self._build_pattern()
        marked = combine_masks(row_mask(4), diagonal_mask("\\"))
        assert pattern.matches(self._card_with_marks(marked))

    def test_matches_when_both_rows_and_diagonal_complete(self):
        # Edge case: AnyPattern doesn't care that BOTH inner options ended
        # up true — one is enough, and having both shouldn't break anything.
        pattern = self._build_pattern()
        marked = combine_masks(row_mask(0), row_mask(4), diagonal_mask("\\"))
        assert pattern.matches(self._card_with_marks(marked))

    def test_does_not_match_diagonal_alone(self):
        # Diagonal complete, but neither row — the AnyPattern branch of
        # AllPattern fails, so the whole thing should fail.
        pattern = self._build_pattern()
        marked = diagonal_mask("\\")
        assert not pattern.matches(self._card_with_marks(marked))

    def test_does_not_match_row_alone(self):
        # Row 0 complete, but no diagonal — the MaskPattern branch of
        # AllPattern fails this time.
        pattern = self._build_pattern()
        marked = row_mask(0)
        assert not pattern.matches(self._card_with_marks(marked))

    def test_does_not_match_wrong_row_plus_diagonal(self):
        # Row 2 (not one of the two allowed rows) plus the diagonal — a
        # row IS complete, but AnyPattern only accepts row 0 or row 4.
        pattern = self._build_pattern()
        marked = combine_masks(row_mask(2), diagonal_mask("\\"))
        assert not pattern.matches(self._card_with_marks(marked))

    def test_does_not_match_nothing_complete(self):
        pattern = self._build_pattern()
        assert not pattern.matches(self._card_with_marks(blank_mask()))
