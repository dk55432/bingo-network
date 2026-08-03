import uuid

class BingoCard:
    
    def __init__(self, 
                 card_id, 
                 player_id,
                 grid
                 ):
        if card_id is None:
            self.card_id = str(uuid.uuid4())
        else:
            self.card_id = card_id
        self.player_id = player_id
        self.grid = grid
        self.marked = [
            [False] * 5 for _ in range(5)
        ]
        # Mark the Free Space
        self.marked[2][2] = True
    
    def is_marked(self, row, col):
        return self.marked[row][col]
    
    def get_number(self, row, col):
        return self.grid[row][col]    
        
    def __str__(self):
        lines = []
        for row in range(5):
            line = ""
            for col in range(5):
                line += f"{self.grid[row][col]:2}"
                if self.is_marked(row, col):
                    line += "* "
                else:
                    line += "  "
            lines.append(line)
        return "\n".join(lines)
        
        
    def mark_number(self, number):
        for row in range(5):
            for col in range(5):
                if self.grid[row][col] == number:
                    self.marked[row][col] = True
                    return True
        return False
