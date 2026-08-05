from player import Player
from enum import Enum

class GameStatus(Enum):
    SETUP = "setup"
    IN_PROGRESS = "in_progress"
    GAME_OVER = "game_over"


class GameState:
    def __init__(self):
        self.status = GameStatus.SETUP
        self.called_numbers = []
        # self.called_set = set()
        self.current_number: str | None = None
        self.players: dict[str, Player] = {}
        
    def __str__(self):
        result = f"status: {self.status}\n"
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
        print("set_game_status: Setting status to: " + str(status))
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
    def call_number(self, number):
        if self.has_called(number):
            print("game.call_number: number "+str(number)+" is already called, returning.")
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
