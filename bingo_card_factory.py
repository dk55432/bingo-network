from bingo_card import BingoCard
from player import Player

def create_test_card(player: Player, whichGrid: int):
    grid1 = [
        ["B1",  "I16", "N31", "G46", "O61"],
        ["B2",  "I17", "N32", "G47", "O62"],
        ["B3",  "I18", "FREE", "G48", "O63"],
        ["B4",  "I19", "N34", "G49", "O64"],
        ["B5",  "I20", "N35", "G50", "O65"],
    ]
    grid2 = [
        ["B6",  "I16", "N31", "G46", "O61"],
        ["B2",  "I17", "N32", "G47", "O62"],
        ["B3",  "I18", "FREE", "G48", "O63"],
        ["B4",  "I19", "N34", "G49", "O64"],
        ["B5",  "I20", "N35", "G50", "O65"],
    ]
    grid3 = [
        ["B7",  "I16", "N31", "G46", "O61"],
        ["B2",  "I17", "N32", "G47", "O62"],
        ["B3",  "I18", "FREE", "G48", "O63"],
        ["B4",  "I19", "N34", "G49", "O64"],
        ["B5",  "I20", "N35", "G50", "O65"],
    ]
    count = len(player.cards)
    if 1 == whichGrid:
        _grid = grid1
    elif 2 == whichGrid:
        _grid = grid2
    else:
        _grid = grid3
    _card_id = str(count + 1)
    return BingoCard(
        card_id=_card_id,
        player_id=None,
        grid=_grid
    )