from player import Player

class GameState:
    def __init__(self):
        self.called_numbers = []
        self.current_number: str | None = None
        self.players: list[Player] = []
        
    def __str__(self):
        result = f"Players: {len(self.players)}\n"
        for player in self.players:
            result += str(player) + "\n"
        # TODO: called_numbers, current_number
        return result
        
    def record_called_number(self, number):
        self.current_number = number
        self.called_numbers.append(number)

    def add_player(self, player: Player):
        self.players.append(player)
        
    def get_player(self, player_id: str):
        for player in self.players:
            if player.player_id == player_id:
                return player
        return None
        
    def find_winners(self):
        winners = []
        for player in self.players:
            for card in player.cards:
                if card.has_bingo():
                    winners.append(
                        (player, card)
                    )
        return winners
        
def canonicalize_number(value):
    value = value.strip().upper()
    if value.isdigit():
        number = int(value)
        # TODO: Handle numbers 1-75, not just 46-60.
        if 46 <= number <= 60:
            return f"G{number}"
    return value
