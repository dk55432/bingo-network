from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import json
import uuid
from datetime import datetime
from connection_manager import ConnectionManager
from game import GameState, canonicalize_number
from player import Player
from bingo_card_factory import create_test_card

app = FastAPI()
manager = ConnectionManager()
game = GameState()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def join_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        # name="player.html"
        # name="join.html"
        name="landing.html"
    )

@app.get("/join")
async def join_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="join.html"
    )
    
@app.get("/player")
async def player_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="player.html"
    )
     
@app.get("/host")
async def host(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="host.html"
    )   
    
    
def disconnect_player(websocket):
    if websocket in manager.active_connections:
        manager.disconnect(websocket)
        
    for player in game.players.values():
        if player.websocket == websocket:
            player.connected = False
            player.websocket = None
            print(f"{player.display_name} disconnected")
            break
        
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print(f"Clients connected: {len(manager.active_connections)}")

    try:
        while True:

            message = await websocket.receive_text()
            print("Received:", message)
            await websocket.send_text(
                json.dumps({
                    "type": "history",
                    "called_numbers": game.called_numbers
                })
            )

            try:
                data = json.loads(message)

            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message!r}")
                continue
            
            if data["type"] == "join":
                player = Player(
                    player_id = str(uuid.uuid4()),
                    display_name = data["display_name"],
                    websocket = websocket,
                    connected_at = datetime.now()
                )
                # TODO: At some point we'll want to store mapping websocket -> player
                
                # TODO: eventually let users enter/scan their own cards.  For now, use test cards 1-3.
                player.add_card(create_test_card(player.display_name, "1"))
                game.add_player(player)

                await websocket.send_text(
                    json.dumps({
                        "type": "joined",
                        "player_id": player.player_id
                    })
                )
                await websocket.send_text(
                    json.dumps({
                        "type": "card",
                        "card": player.cards[0].to_dict()
                    })
                )                
                # Debug
                print("Post-join: Players:")
                for player in game.players.values():
                    print(
                        player.display_name,
                        player.connected,
                        player.websocket is not None
                    )
                
            elif data["type"] == "submit_number":
                number = canonicalize_number(data["value"])
                print("submit_number: just received number: " + str(number))
                result = game.call_number(number)
                print("submit_number: broadcasting Hx: "+str(game.called_numbers))
                await manager.broadcast(
                    json.dumps({
                        "type": "history",
                        "called_numbers": game.called_numbers
                    })
                )
                
                winners = result["winners"]
                updated_cards = result["updated_cards"]
                
                print("UPDATED CARDS:")
                for update in updated_cards:
                    print(update)
                    
                for update in updated_cards:
                    player = game.players[update["player_id"]]
                    if player.websocket is None:
                        continue
                    await manager.send_to_player(
                        player.websocket,
                        json.dumps({
                            "type": "card_update",
                            "card": update["card"]
                        })
                    )
        
                print("WINNERS:", winners)

                await manager.broadcast(
                    json.dumps({
                        "type": "number_called",
                        "number": number
                    })
                )

                if winners:
                    await manager.broadcast(
                        json.dumps({
                            "type": "winner",
                            "winners": winners
                        })
                    )
                    print("WINNER EVENT SENT")        
        
            elif data["type"] == "reconnect":
                player_id = data["player_id"]
                player = game.players.get(player_id)
                if player:
                    player.websocket = websocket
                    player.connected = True
                    await websocket.send_text(
                        json.dumps({
                            "type": "reconnected"
                        })
                    )
                    # Re-send history only to this reconnected player
                    await websocket.send_text(
                        json.dumps({
                            "type": "history",
                            "called_numbers": game.called_numbers
                        })
                    )

                    # DAVE: I keep getting cards[0] error, cards is null
                    # Clean this issue up when refactoring to support multi-cards.
                    if len(player.cards) == 0:
                        await websocket.send_text(
                            json.dumps({
                                "type": "card",
                                "card": None
                            })
                        ) 
                    else:
                        await websocket.send_text(
                            json.dumps({
                                "type": "card",
                                "card": player.cards[0].to_dict()
                            })
                        ) 
                    # Debug
                    print("Post-reconnect: Players:")
                    for player in game.players.values():
                        print(
                            player.display_name,
                            player.connected,
                            player.websocket is not None
                        )

            elif data["type"] == "leave":
                disconnect_player(websocket)
                await websocket.send_text(
                    json.dumps({"type": "left"})
                )
            
            elif data["type"] == "start_new_game":
                print("start_new_game")
                game.start_new_game()
                await manager.broadcast(
                    json.dumps({
                        "type": "new_game"
                    })
                )
                await broadcast_history(game)
                await broadcast_cards(game)
                
            # TODO: This goes away when we implement OCR scanning / card inputs.
            elif data["type"] == "load_test_card":
                player_id = data["player_id"]
                number = data["number"]
                player = game.get_player(player_id)
                print("In load_test_card block, will load grid "+str(number))
                player.dispose_cards() # DAVE: Do I need to do this?
                player.add_card(create_test_card("test card", number))
                await websocket.send_text(
                    json.dumps({
                        "type": "card",
                        "card": player.cards[0].to_dict()
                    })
                )     
                           
            elif data["type"] == "dispose_cards":
                print("Entering disposeCards block")
                player_id = data["player_id"]
                player = game.get_player(player_id)
                print("disposeCards for player "+ str(player))
                # DAVE: null player when server restarts but browser doesn't.
                if player is not None:
                    player.dispose_cards()
                print("after disposeCards for player "+ str(player))
                await websocket.send_text(
                    json.dumps({
                        "type": "card",
                        "card": []
                    })
                )     
                

    except WebSocketDisconnect:
        disconnect_player(websocket)

        print(f"Clients connected: {len(manager.active_connections)}")
        # Debug
        print("Post-disconnect: Players:")
        for player in game.players.values():
            print(
                player.display_name,
                player.connected,
                player.websocket is not None
            )

async def broadcast_history(game):
    await manager.broadcast(
        json.dumps({
            "type": "history",
            "called_numbers": game.called_numbers
        })
    )

async def broadcast_cards(game):
    for player in game.players.values():
        if player.cards is not None and len(player.cards) > 0:
            await manager.broadcast(
                json.dumps({
                    "type": "card",
                    "card": player.cards[0].to_dict()
                })
            )     
        else:
            await manager.broadcast(
                json.dumps({
                    "type": "card",
                    "card": []
                })
            )     
            