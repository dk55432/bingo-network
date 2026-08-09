import uuid

class BingoCard:
    
    def __init__(self, 
                 card_id, 
                #  display_name,
                 player_id,
                 grid
                 ):
        if card_id is None:
            self.card_id = str(uuid.uuid4())
        else:
            self.card_id = card_id
        # self.display_name = display_name
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
    
    
    # Win-checking used to live here as a hardcoded row/col/diagonal check.
    # It's now handled by the WinningPattern classes (see winning_pattern.py)
    # instead — Game.call_number() calls self.winning_pattern.matches(card),
    # where winning_pattern is whichever pattern was selected for this game
    # (see patterns_config.txt / patterns_parser.py). This lets a game use
    # any configured pattern (postage stamp, blackout, etc.), not just
    # standard bingo, without BingoCard needing to know about any of them.


    def to_dict(self):
        return {
            "card_id": self.card_id,
            # "display_name": self.display_name,
            "grid": self.grid,
            "marked": self.marked
        }