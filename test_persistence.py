"""Persistence round-trip tests for game state (GameStore + GameManager).

Pure stdlib + patterns_parser — no torch/tensor deps, so this runs in a
fast CI job separate from the CNN regression suite.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from bingo_card import BingoCard
from game import GameManager, GameStatus, canonicalize_number
from game_store import GameStore
from player import Player
import patterns_parser


@pytest.fixture()
def patterns():
    cfg = Path(__file__).resolve().parent / "patterns_config.txt"
    return patterns_parser.load_patterns_from_file(str(cfg))


def _make_rich_manager(patterns):
    """A game with two players, a marked card, a winning row, waiting-room
    entry, and IN_PROGRESS status."""
    gm = GameManager()
    game = gm.create_game()
    game.set_winning_pattern("standard-bingo", patterns["standard-bingo"])

    alice = Player(player_id="p-alice", display_name="alice", websocket=None)
    grid_a = [
        ["B1", "I16", "N31", "G46", "O61"],
        ["B6", "I17", "N32", "G47", "O62"],
        ["B11", "I18", "FREE", "G48", "O63"],
        ["B12", "I19", "N34", "G49", "O64"],
        ["B13", "I20", "N35", "G50", "O65"],
    ]
    alice.add_card(BingoCard(card_id="a1", player_id="p-alice", grid=grid_a))
    game.add_player(alice)

    bob = Player(player_id="p-bob", display_name="bob", websocket=None)
    bob.add_card(BingoCard(card_id="b1", player_id="p-bob",
                           grid=[[0] * 5 for _ in range(5)]))
    game.add_player(bob)

    # Fill alice's top row so standard-bingo should win after restore.
    for value in ["1", "16", "31", "46", "61"]:
        game.call_number(canonicalize_number(value))
    game.set_game_status(GameStatus.IN_PROGRESS)
    game.waiting_room["pend-1"] = {"display_name": "waiting-dave",
                                   "websocket": None}
    return gm


def test_snapshot_restore_roundtrip(tmp_path, patterns):
    gm = _make_rich_manager(patterns)
    game = next(iter(gm.games.values()))

    store = GameStore(str(tmp_path / "state.db"))
    store.save_all(gm.snapshot())

    restored = GameManager().restore_from(store.load_all(), patterns)
    g = restored.get_game(game.game_id)
    assert g is not None
    assert g.status == GameStatus.IN_PROGRESS
    assert g.called_numbers == ["B1", "I16", "N31", "G46", "O61"]
    assert g.current_number == "O61"
    assert g.winning_pattern_name == "standard-bingo"
    assert g.winning_pattern is not None
    assert (abs((g.last_activity - game.last_activity).total_seconds())
            < 5)

    # waiting room: names survive, websockets never persisted.
    assert g.waiting_room["pend-1"]["display_name"] == "waiting-dave"
    assert g.waiting_room["pend-1"]["websocket"] is None

    # players + cards + marks restored with stable ids.
    alice = g.get_player("p-alice")
    assert alice is not None
    assert alice.display_name == "alice"
    assert len(alice.cards) == 1
    card = alice.cards[0]
    assert card.card_id == "a1"
    assert card.grid[0] == ["B1", "I16", "N31", "G46", "O61"]
    assert card.marked[0] == [True] * 5
    assert card.marked[2][2] is True  # free space

    # restored board is actually winnable per the restored pattern.
    winners = g.find_winners()
    assert any(w[1] == "alice" for w in winners)
    assert not any(w[1] == "bob" for w in winners)


def test_snapshot_is_json_serializable(patterns):
    gm = _make_rich_manager(patterns)
    json.dumps(gm.snapshot())


def test_save_all_replaces_table_and_propagates_deletions(tmp_path, patterns):
    store = GameStore(str(tmp_path / "state.db"))
    gm = _make_rich_manager(patterns)
    game = next(iter(gm.games.values()))
    store.save_all(gm.snapshot())
    assert len(store.load_all()) == 1

    # A game removed from memory disappears from the store on the next save.
    blob = gm.snapshot()[game.game_id]
    gm.games.clear()
    store.save_all(gm.snapshot())  # empty snapshot -> no-op, nothing deleted
    assert len(store.load_all()) == 1

    store.save_all({"ghost": blob})
    assert set(store.load_all()) == {"ghost"}


def test_upsert_single_game(tmp_path, patterns):
    store = GameStore(str(tmp_path / "state.db"))
    gm = _make_rich_manager(patterns)
    game = next(iter(gm.games.values()))
    blob = game.to_persistable()
    store.upsert(game.game_id, blob)
    loaded = store.load_all()
    assert set(loaded) == {game.game_id}
    assert loaded[game.game_id]["status"] == "in_progress"

    blob["status"] = "game_over"
    store.upsert(game.game_id, blob)
    assert store.load_all()[game.game_id]["status"] == "game_over"


def test_corrupt_blob_is_skipped(tmp_path, patterns):
    db = str(tmp_path / "state.db")
    store = GameStore(db)
    gm = _make_rich_manager(patterns)
    game = next(iter(gm.games.values()))
    store.save_all(gm.snapshot())
    store.close()

    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM games")
    conn.execute(
        "INSERT INTO games (game_id, blob, updated_at) VALUES (?, ?, ?)",
        ("bad", "{not json", "now"),
    )
    conn.commit()
    conn.close()

    store2 = GameStore(db)
    assert store2.load_all() == {}
    store2.close()


def test_restore_unknown_pattern_disables_wins(patterns):
    blob = {"gx": {"game_id": "gx", "status": "in_progress",
                   "winning_pattern_name": "no-such-pattern",
                   "players": {}}}
    restored = GameManager().restore_from(blob, patterns)
    g = restored.get_game("gx")
    assert g.status == GameStatus.IN_PROGRESS
    assert g.winning_pattern_name == "no-such-pattern"
    assert g.winning_pattern is None
    # call_number stays safe: records the call, finds no winners.
    result = g.call_number("B1")
    assert result == {"winners": [], "updated_cards": []}
    assert g.called_numbers == ["B1"]


def test_restore_corrupt_status_falls_back_to_setup(patterns):
    blob = {"gy": {"game_id": "gy", "status": "definitely-not-a-status",
                   "players": {}}}
    restored = GameManager().restore_from(blob, patterns)
    assert restored.get_game("gy").status == GameStatus.SETUP