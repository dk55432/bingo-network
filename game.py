from player import Player

class GameState:
    def __init__(self):
        self.called_numbers = []
        self.called_set = set()
        self.current_number: str | None = None
        self.players: dict[str, Player] = {}
        
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
        self.players[player.player_id] = player
                
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
    
    
    def call_number(self, number):
        
        if number in self.called_set:
            return {
                "winners": [],
                "updated_cards": []
            }
    
        self.current_number = number
        self.called_numbers.append(number)
        self.called_set.add(number)

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
                        "card_id": card.card_id
                    })
        return {
            "winners": winners,
            "updated_cards": updated_cards
        }    
                    
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
