"""
Integration tests for winning_pattern.py — the real pipeline: a Game
marks real cards via call_number(), and the pattern classes check the
result.

This file needs the full project import chain (Game, Player,
bingo_card_factory, and — transitively, via player.py — fastapi). That's
expected: it's specifically testing that the pieces work together, not
testing any one class in isolation. If this file's imports fail but
test_winning_pattern_unit.py's don't, that's actually useful information:
it tells you the problem is in the game/player/card side of the project,
not in winning_pattern.py itself.

For dependency-free tests of MaskPattern/AnyPattern/AllPattern alone, see
test_winning_pattern_unit.py instead.

Run with: pytest test_winning_pattern_integration.py -v
"""

from datetime import datetime

import pytest

from pattern_helpers import standard_bingo_pattern, both_diagonals_pattern
from bingo_card_factory import create_test_card
from game import Game
from player import Player


@pytest.fixture
def game():
    return Game()


@pytest.fixture
def player(game):
    p = Player(
        player_id="test-player",
        display_name="Tester",
        websocket=None,
        connected_at=datetime.now()
    )
    game.add_player(p)
    return p


@pytest.fixture
def two_cards(player):
    """Two fresh test cards (#1 and #2), freshly attached to `player`.

    Fresh per test by construction (function-scoped fixture) — this is
    what fixes the state-leakage bug the original script had, where a
    second test's create_cards() call appended onto the same list the
    first test had already marked up, rather than starting clean.
    """
    cards = [create_test_card(player, 1), create_test_card(player, 2)]
    player.cards = cards
    return cards


class TestPatternsWithRealGameFlow:
    def test_any_pattern_matches_a_completed_column(self, game, two_cards):
        # Test card #1's B column is B1-B5, by construction — calling all
        # five completes it.
        pattern = standard_bingo_pattern()

        for number in ["B1", "B2", "B3", "B4", "B5"]:
            game.call_number(number)

        assert pattern.matches(two_cards[0])
        assert not pattern.matches(two_cards[1])

    def test_all_pattern_matches_both_diagonals(self, game, two_cards):
        # Test card #1's two diagonals (minus the free center square) sit
        # at these eight numbers, by construction.
        pattern = both_diagonals_pattern()

        for number in ["B1", "B5", "I17", "I19", "G47", "G49", "O61", "O65"]:
            game.call_number(number)

        assert pattern.matches(two_cards[0])
        assert not pattern.matches(two_cards[1])

    def test_all_pattern_does_not_match_with_only_one_diagonal_complete(self, game, two_cards):
        # A stricter check than the original script had: complete only
        # ONE diagonal and confirm AllPattern correctly still says no.
        pattern = both_diagonals_pattern()

        for number in ["B1", "I17", "G49", "O65"]:  # only the "\" diagonal
            game.call_number(number)

        assert not pattern.matches(two_cards[0])
