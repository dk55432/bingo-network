from bingo_card import BingoCard


def create_test_card(card_id="test-card-1"):

    grid = [
        ["B1",  "I16", "N31", "G46", "O61"],
        ["B2",  "I17", "N32", "G47", "O62"],
        ["B3",  "I18", "FREE", "G48", "O63"],
        ["B4",  "I19", "N34", "G49", "O64"],
        ["B5",  "I20", "N35", "G50", "O65"],
    ]

    return BingoCard(
        card_id=card_id,
        player_id=None,
        grid=grid
    )