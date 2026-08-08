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


class ConstraintPattern(WinningPattern):
    """
    Counts how many of a given mask's required cells are marked, and
    compares that count against a threshold. AtLeast(mask, n) and
    AtMost(mask, n) from the config file both become one of these —
    same counting logic, just a different comparison direction — rather
    than needing a separate class per row/column/direction combination.
    """

    AT_LEAST = "at_least"
    AT_MOST = "at_most"

    def __init__(self, name: str, mask, comparison: str, n: int):
        if comparison not in (self.AT_LEAST, self.AT_MOST):
            raise ValueError(
                f"Unknown comparison {comparison!r}; expected "
                f"{self.AT_LEAST!r} or {self.AT_MOST!r}"
            )
        self.name = name
        self.mask = mask
        self.comparison = comparison
        self.n = n

    @classmethod
    def at_least(cls, name: str, mask, n: int) -> "ConstraintPattern":
        return cls(name, mask, cls.AT_LEAST, n)

    @classmethod
    def at_most(cls, name: str, mask, n: int) -> "ConstraintPattern":
        return cls(name, mask, cls.AT_MOST, n)

    def _count_marked(self, card: BingoCard) -> int:
        count = 0
        for row in range(5):
            for col in range(5):
                if self.mask[row][col] and card.marked[row][col]:
                    count += 1
        return count

    def matches(self, card: BingoCard) -> bool:
        count = self._count_marked(card)
        if self.comparison == self.AT_LEAST:
            return count >= self.n
        else:  # AT_MOST
            return count <= self.n
    