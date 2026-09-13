from player import Player
from bingo_card import BingoCard
from enum import Enum
import logging
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GameStatus(Enum):
    SETUP = "setup"
    IN_PROGRESS = "in_progress"
    GAME_OVER = "game_over"

class GameStatusString(Enum):
    SETUP = "🟢 Waiting for players"
    IN_PROGRESS = "🔴 Game in progress"
    GAME_OVER = "🏁 Game finished"

class GameManager:
    # How long a game with no host and no connected players sits around
    # before it's considered abandoned rather than mid-reconnect.
    DEFAULT_GRACE_PERIOD_MINUTES = 30

    def __init__(self):
        self.games: dict[str, Game] = {}
     
    def create_game(self):
        game = Game()
        game.game_id = str(uuid.uuid4())
        self.games[game.game_id] = game
        logger.info("create_game: game_id "+game.game_id)
        return game

    def get_game(self, game_id):
        return self.games.get(game_id)

    def delete_game(self, game_id):
        self.games.pop(game_id, None)

    def cleanup_empty_games(self, grace_period_minutes=DEFAULT_GRACE_PERIOD_MINUTES):
        """Remove games that have no host attached, no currently-connected
        players, and haven't seen any activity within the grace period.

        Deliberately NOT based on len(game.players) == 0 — players stay in
        that dict after disconnecting (only an explicit "leave" removes
        them), so that would only ever catch games nobody ever joined.
        Checking player.connected instead reflects who's actually here
        right now, and the grace period protects against sweeping up a
        game mid-transient-disconnect (e.g. a host's wifi blipping).
        """
        now = datetime.now()
        grace = timedelta(minutes=grace_period_minutes)

        empty_ids = [
            game_id for game_id, game in self.games.items()
            if game.host_websocket is None
            and not any(p.connected for p in game.players.values())
            and not game.waiting_room
            and (now - game.last_activity) >= grace
        ]
        for game_id in empty_ids:
            logger.info("cleanup_empty_games: removing inactive game "+game_id)
            del self.games[game_id]
        return empty_ids

    def snapshot(self):
        """Plain-dict snapshot of every live game, for the persistence
        layer. The in-memory manager is the source of truth."""
        return {
            game_id: game.to_persistable()
            for game_id, game in self.games.items()
        }

    def restore_from(self, persisted, patterns_config):
        """Rebuild games from {game_id: persisted dict}.

        patterns_config maps pattern names to WinningPattern instances
        (main.py's WINNING_PATTERNS). Games whose pattern name is no longer
        configured are restored with winning_pattern=None — call_number()
        then treats them as "nobody can win yet" rather than erroring.
        """
        for game_id, blob in persisted.items():
            game = Game.from_persisted(blob, patterns_config)
            self.games[game_id] = game
        if persisted:
            logger.info(
                "GameManager: restored %d game(s)", len(persisted)
            )
        return self
    
    
class Game:
    def __init__(self):
        self.status = GameStatus.SETUP
        self.game_id = ""
        self.called_numbers = []
        # self.called_set = set()
        self.current_number: str | None = None
        self.players: dict[str, Player] = {}
        self.host_websocket = None  # set once a host creates/reconnects to this game
        self.last_activity = datetime.now()
        # People who tried to join while status != SETUP. Keyed by a
        # throwaway id (not a player_id — they aren't real players yet).
        # Promoted into real players once setup_new_game() runs.
        self.waiting_room: dict[str, dict] = {}
        # Set via set_winning_pattern() — main.py assigns a default
        # ("standard-bingo", loaded from patterns_config.txt) immediately
        # after creating a Game, so in normal operation this is never
        # actually None by the time call_number() runs. It's None here
        # only as a safe default for callers (e.g. tests) that construct
        # a bare Game() without going through that wiring — call_number()
        # treats "no pattern set" as "nobody can win yet" rather than
        # crashing.
        self.winning_pattern = None
        self.winning_pattern_name = None

    def set_winning_pattern(self, name: str, pattern):
        self.winning_pattern_name = name
        self.winning_pattern = pattern

    def to_persistable(self):
        """Plain-dict snapshot of the game's durable state.

        Connection state (host_websocket, player websockets / connected
        flags) is transient and never persisted; waiting-room entries lose
        their websocket too — a waiter whose browser survived still
        re-joins, and setup_new_game() skips entries whose websocket is no
        longer around.
        """
        return {
            "game_id": self.game_id,
            "status": self.status.value,
            "called_numbers": list(self.called_numbers),
            "current_number": self.current_number,
            "winning_pattern_name": self.winning_pattern_name,
            "last_activity": self.last_activity.isoformat(timespec="seconds"),
            "waiting_room": {
                pending_id: {"display_name": entry["display_name"]}
                for pending_id, entry in self.waiting_room.items()
            },
            "players": {
                player_id: player.to_persistable()
                for player_id, player in self.players.items()
            },
        }

    @classmethod
    def from_persisted(cls, blob, patterns_config):
        """Rebuild a Game from a to_persistable() dict (see
        GameManager.restore_from)."""
        game = cls()
        game.game_id = blob.get("game_id", "")
        try:
            game.status = GameStatus(blob["status"])
        except (KeyError, ValueError):
            game.status = GameStatus.SETUP
        game.called_numbers = list(blob.get("called_numbers") or [])
        game.current_number = blob.get("current_number")
        game.winning_pattern_name = blob.get("winning_pattern_name")
        name = game.winning_pattern_name
        if name:
            game.winning_pattern = patterns_config.get(name)
            if game.winning_pattern is None:
                logger.warning(
                    "game %s references unknown pattern %r — wins disabled",
                    game.game_id, name,
                )
        try:
            game.last_activity = datetime.fromisoformat(blob["last_activity"])
        except (KeyError, TypeError, ValueError):
            game.last_activity = datetime.now()
        game.waiting_room = {
            pending_id: {
                "display_name": entry.get("display_name", ""),
                "websocket": None,
            }
            for pending_id, entry in (blob.get("waiting_room") or {}).items()
        }
        for player_id, pd in (blob.get("players") or {}).items():
            connected_at = None
            raw = pd.get("connected_at")
            if raw:
                try:
                    connected_at = datetime.fromisoformat(raw)
                except (TypeError, ValueError):
                    connected_at = None
            player = Player(
                player_id=pd.get("player_id", player_id),
                display_name=pd.get("display_name", ""),
                connected=False,
                websocket=None,
                connected_at=connected_at,
            )
            for card_d in pd.get("cards") or []:
                grid = card_d.get("grid") or [[0] * 5 for _ in range(5)]
                card = BingoCard(
                    card_id=card_d.get("card_id"),
                    player_id=card_d.get("player_id", player.player_id),
                    grid=grid,
                )
                marked = card_d.get("marked")
                if marked:
                    card.marked = marked
                player.add_card(card)
            game.add_player(player)
        return game

    def touch(self):
        """Call whenever someone connects or disconnects (join, reconnect,
        host_reconnect, or either side leaving), so cleanup_empty_games
        can tell a truly-abandoned game apart from one that's just
        between connections."""
        self.last_activity = datetime.now()
        
    def __str__(self):
        result = f"game_id: {self.game_id}\n"
        result += f"status: {self.status}\n"
        result += f"Players: {len(self.players)}\n"
        for player in self.players.values():
            result += str(player) + "\n"
        result += f"current_number: {self.current_number}\n"
        result += f"called_numbers: {self.called_numbers}\n"
        return result
        
    def record_called_number(self, number):
        self.current_number = number
        self.called_numbers.append(number)
    
    def clear_called_numbers(self):
        self.called_numbers.clear()
        
    def has_called(self, number):
        return number in self.called_numbers
        
    def clear_current_number(self):
        self.current_number = None
        
    def set_current_number(self, number):
        self.current_number = number
        
    def get_current_number(self):
        return self.current_number

    def add_player(self, player: Player):
        self.players[player.player_id] = player
        
    def remove_player(self, player_id: str):
        self.players.pop(player_id, None)
                
    def get_player(self, player_id: str):
        return self.players.get(player_id)
    
    def get_cards_for_player(self, player_id):
        player = self.players.get(player_id)
        if player is None:
            return []
        return [
            card.to_dict()
            for card in player.cards
        ]
    
    # It's expected that the code in main.py will broadcast the status change.
    def set_game_status(self, status: Enum):
        logger.debug("set_game_status: Setting status to: " + str(status))
        self.status = status
        return {"type": "game_status", "status": status}
        
    
    def setup_new_game(self):
        self.clear_called_numbers()
        self.clear_current_number()
        for player in self.players.values():
            player.dispose_cards()
        self.set_game_status(GameStatus.SETUP)
        
    # Returns a dict of 2 arrays: 
    #   - winners: array of dicts { player_id, display_name, card_id }
    #   - updated_cards: array of dicts { player_id, card }
    #
    # TODO: split it into call_number() calling:
    # _record_number()
    # _mark_cards()
    # _find_new_winners()
    #
    # Also, create a python GameEvent class; Game.call_numbers() returns a GameEvent
    def call_number(self, number):
        if self.has_called(number):
            logger.info("game.call_number: number "+str(number)+" is already called, returning.")
            return {
                "winners": [],
                "updated_cards": []
            }
    
        self.record_called_number(number)

        winners = []
        updated_cards = []

        for player in self.players.values():
            for card in player.cards:
                card.mark_number(number)
                updated_cards.append({
                    "player_id": player.player_id,
                    "card": card.to_dict()
                })

                if self.winning_pattern is not None and self.winning_pattern.matches(card):
                    winners.append({
                        "player_id": player.player_id,
                        "display_name": player.display_name,
                        "card_id": card.card_id
                    })
        return {
            "winners": winners,
            "updated_cards": updated_cards
        }    
                    
    def find_winners(self):
        winners = []
        if self.winning_pattern is None:
            return winners
        for player in self.players.values():
            display_name = player.display_name
            for card in player.cards:
                if self.winning_pattern.matches(card):
                    winners.append(
                        (player, display_name, card)
                    )
        return winners
        
def canonicalize_number(value):
    value = value.strip().upper()
    if value.isdigit():
        number = int(value)
        if 1 <= number <= 15:
            return f"B{number}"
        elif 16 <= number <= 30:
            return f"I{number}"
        elif 31 <= number <= 45:
            return f"N{number}"
        elif 46 <= number <= 60:
            return f"G{number}"
        else:
            return f"O{number}"
    return value
