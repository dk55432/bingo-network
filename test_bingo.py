from datetime import datetime

from game import GameState
from player import Player
from bingo_card_factory import create_test_card


game = GameState()


player = Player(
    player_id="test-player",
    # name="test",
    display_name="Tester",
    websocket=None,
    connected_at=datetime.now()
)


player.add_card(
    create_test_card()
)


game.add_player(player)


print("Before calls:")
print(player)

# Horizontal bingo, row 1
# game.call_number("B1")
# game.call_number("I16")
# game.call_number("N31")
# game.call_number("G46")
# game.call_number("O61")

# Vertical bingo, col 1
game.call_number("B1")
game.call_number("B2")
game.call_number("B3")
game.call_number("B4")
game.call_number("B5")

# Diagonal bingo starting at [1],[1]
# game.call_number("B1")
# game.call_number("I17")
# game.call_number("FREE")
# game.call_number("G49")
# game.call_number("O65")

# Diagonal bingo starting at [5],[1]
# game.call_number("B5")
# game.call_number("I19")
# game.call_number("FREE")
# game.call_number("G47")
# game.call_number("O61")

# game.call_number("B1")
# game.call_number("B1")
# print ("len of called_numbers is "+str(len(game.called_numbers)))

print("After calls:")
print(player)
print(player.cards[0].to_dict())

print(
    "Bingo?",
    player.cards[0].has_bingo()
)