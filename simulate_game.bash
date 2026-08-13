
# To run:   bash simulate_game.bash

export GAME_ID="febddacc-e771-46be-80d1-4bbccc78de8d"
export NUM_PLAYERS=200
export CARDS_PER_PLAYER=100
export CARD_FIXTURES_JSON='[
  [
    [1,16,31,46,61],
    [2,17,32,47,62],
    [3,18,null,48,63],
    [4,19,34,49,64],
    [5,20,35,50,65]
  ]
]'
export WS_URL="ws://192.168.1.234:8000/ws"
export API_BASE="http://192.168.1.234:8000"

node simulate_game.mjs
