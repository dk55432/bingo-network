from player import Player
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

                if card.has_bingo():
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
        for player in self.players.values():
            display_name = player.display_name
            for card in player.cards:
                if card.has_bingo():
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
