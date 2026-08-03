from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import json
import uuid
from datetime import datetime
from connection_manager import ConnectionManager
from game import GameState, canonicalize_number
from player import Player

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
                
                game.add_player(player)
                print("Game is now: "+str(game))
                await websocket.send_text(
                    json.dumps({
                        "type": "joined",
                        "player_id": player.get_player_id()
                    })
                )
                
            elif data["type"] == "submit_number":
                number = canonicalize_number(data["value"])
                event = {
                    "type": "number_called",
                    "number": number
                }
                
                # TODO: Right now the server is responsible for these 3 commands. As the game logic grows, we'll want GameState to own more of the "what happens when a number is called" workflow.  That keeps the WebSocket handler focused on networking rather than game rules.
                accepted = game.call_number(number)
                if accepted:
                    await manager.broadcast(
                        json.dumps(event)
                    )  
                    print(game.called_numbers)

            elif data["type"] == "reconnect":
                player = game.get_player(data["player_id"])
                if player:
                    player.connect(websocket)
                    await websocket.send_text(json.dumps({
                        "type": "reconnected"
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "reconnect_failed"
                    }))
                    # print(f"{player.display_name} reconnected")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Clients connected: {len(manager.active_connections)}")
    
    