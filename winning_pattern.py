from abc import ABC, abstractmethod
from bingo_card import BingoCard

class WinningPattern(ABC):

    @abstractmethod
    def matches(self, card):
        """
        Return True if this card satisfies
        this winning pattern.
        """
        pass
    
class MaskPattern(WinningPattern):
    def __init__(self, name, mask):
        self.name = name
        self.mask = mask
        
    def matches(self, card):
        for row in range(5):
            for col in range(5):
                if self.mask[row][col]:
                    if not card.marked[row][col]:
                        return False
        return True


class AnyPattern(WinningPattern):
    def __init__(self, patterns: list[WinningPattern]):
        self.patterns = patterns

    def matches(self, card: BingoCard) -> bool:
        for pattern in self.patterns:
            if pattern.matches(card):
                return True
        return False
    

class AllPattern(WinningPattern):
    def __init__(self, patterns: list[WinningPattern]):
        self.patterns = patterns
        
    def matches(self, card: BingoCard) -> bool:
        for pattern in self.patterns:
            if not pattern.matches(card):
                return False
        return True
    