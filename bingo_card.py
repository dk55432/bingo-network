import uuid

class BingoCard:
    
    def __init__(self, 
                 card_id, 
                 display_name,
                 player_id,
                 grid
                 ):
        if card_id is None:
            self.card_id = str(uuid.uuid4())
        else:
            self.card_id = card_id
        self.display_name = display_name
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
                    if not self.marked[row][col]:
                        self.marked[row][col] = True
                        return True
                    return False
        return False    
    
    
    # TODO: Need to accommodate more custom bingo winner layouts like postage-stamp
    def has_bingo(self):
        # check rows
        for row in range(5):
            if all(self.marked[row]):
                return True

        # check columns
        for col in range(5):
            if all(self.marked[row][col] for row in range(5)):
                return True

        # check diagonal 1
        if all(self.marked[i][i] for i in range(5)):
            return True

        # check diagonal 2
        if all(self.marked[i][4-i] for i in range(5)):
            return True

        return False


    def to_dict(self):
        return {
            "card_id": self.card_id,
            "display_name": self.display_name,
            "grid": self.grid,
            "marked": self.marked
        }