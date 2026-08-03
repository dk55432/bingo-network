from bingo_card import BingoCard
from fastapi import WebSocket

class Player:
    def __init__(self,
                 player_id,
                 display_name,
                 websocket=None,
                 connected_at=None):

        self.player_id: str = player_id
        self.display_name: str = display_name
        self.websocket = websocket
        self.cards: list[BingoCard] = []
        self.connected_at = connected_at
    
    def __str__(self):
        result = f"Player: {self.player_id}\n"
        result += f"Display name: {self.display_name}\n"
        result += f"Connected: {self.websocket is not None}\n"
        result += f"Cards: {len(self.cards)}\n"
        
        for card in self.cards:
            result += str(card) + "\n"
        
        result += f"Connected at: {self.connected_at}\n"
        
        return result
    
    def add_card(self, card: BingoCard):
        self.cards.append(card)
        
    def connect(self, websocket: WebSocket):
        self.websocket = websocket
    
    def disconnect(self):
        self.websocket = None
    
    def get_player_id(self):
        return self.player_id
        
    