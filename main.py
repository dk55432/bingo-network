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
                
                player.add_card(create_test_card())
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
                
            elif data["type"] == "submit_number":

                number = canonicalize_number(data["value"])

                event = {
                    "type": "number_called",
                    "number": number
                }

                result = game.call_number(number)
                winners = result["winners"]
                updated_cards = result["updated_cards"]
                
                print("UPDATED CARDS:")
                for update in updated_cards:
                    print(update)
                
                for update in updated_cards:

                    player_id = update["player_id"]

                    player = game.players[player_id]

                    await manager.send_to_player(
                        player.websocket,
                        json.dumps({
                            "type": "card_update",
                            "card": update["card"]
                        })
                    )
    
                print(game.called_numbers)
                print("WINNERS:", winners)

                await manager.broadcast(
                    json.dumps(event)
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

                    await websocket.send_text(
                        json.dumps({
                            "type": "reconnected"
                        })
                    )

                    await websocket.send_text(
                        json.dumps({
                            "type": "card",
                            "card": player.cards[0].to_dict()
                        })
                    ) 
                       
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Clients connected: {len(manager.active_connections)}")
    
    