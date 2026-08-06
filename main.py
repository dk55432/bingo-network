from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import json
import uuid
from datetime import datetime
from connection_manager import ConnectionManager
from game import GameState, GameStatus, canonicalize_number
from player import Player
from bingo_card_factory import create_test_card
import logging

app = FastAPI()
manager = ConnectionManager()
game = GameState()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
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
            logger.debug(f"{player.display_name} disconnected")
            break
        
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info(f"Clients connected: {len(manager.active_connections)}")

    try:
        while True:

            message = await websocket.receive_text()
            logger.info("Received: %s", message)

            try:
                data = json.loads(message)

            except json.JSONDecodeError:
                logger.info(f"Invalid JSON received: {message!r}")
                continue
            
            if data["type"] == "join":
                if game.status != GameStatus.SETUP:
                    logger.info("join: can only join during SETUP")
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "join",
                            "reason": "game_in_progress"
                        })
                    )
                    continue
                    
                player = Player(
                    player_id = str(uuid.uuid4()),
                    display_name = data["display_name"],
                    websocket = websocket,
                    connected_at = datetime.now()
                )
                # TODO: At some point we'll want to store mapping websocket -> player
                
                # TODO: eventually let users enter/scan their own cards.  For now, use test cards 1-3.
                player.add_card(create_test_card(player, 1))
                game.add_player(player)

                await manager.send_to_player( player.websocket,
                    json.dumps({
                        "type": "joined",
                        "player_id": player.player_id
                    })
                )
                await manager.send_to_player( player.websocket,
                    json.dumps({
                        "type": "cards",
                        "cards": cards_to_dict(player.cards)
                    })
                )            
                await manager.send_to_player( player.websocket,
                    json.dumps({
                        "type": "history",
                        "called_numbers": game.called_numbers
                    })
                )
    
                # Debug
                logger.debug("Post-join: Players:")
                for player in game.players.values():
                    print(
                        player.display_name,
                        player.connected,
                        player.websocket is not None
                    )
                
            elif data["type"] == "submit_number":
                if game.status != GameStatus.IN_PROGRESS:
                    logger.info("submit_number: cannot submit number unless status IN_PROGRESS.")
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "submit_number",
                            "reason": "game_not_in_progress"
                        })
                    )
                    continue
                
                number = canonicalize_number(data["value"])
                logger.info("submit_number: just received number: " + str(number))
                result = game.call_number(number)
                logger.debug("submit_number: broadcasting Hx: "+str(game.called_numbers))
                await manager.broadcast(
                    json.dumps({
                        "type": "history",
                        "called_numbers": game.called_numbers
                    })
                )
                
                winners = result["winners"]
                updated_cards = result["updated_cards"]
                
                logger.debug("UPDATED CARDS:")
                for update in updated_cards:
                    logger.debug(update)
                    
                for update in updated_cards:
                    player = game.players[update["player_id"]]
                    if player is None:
                        await manager.send_to_player(
                            websocket,
                            json.dumps({
                                "type": "reconnect_failed"
                            })
                        )
                        continue
                    # await manager.send_to_player( player.websocket,
                    #     json.dumps({
                    #         "type": "cards",
                    #         "cards": cards_to_dict(player.cards)
                    #     })
                    # )            
        
                logger.debug("WINNERS: %s", winners)

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
                    logger.info("WINNER EVENT SENT")        
        
            elif data["type"] == "reconnect":
                player_id = data["player_id"]
                if not player_id:
                    await websocket.send_json({"type":"error", "message":"player_id required"})
                    continue
                logger.info("reconnect: fetched player_id "+player_id)
                player = game.players.get(player_id)
                if player is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "reconnect_failed"
                        })
                    )
                    continue
                player.websocket = websocket
                player.connected = True
                await manager.send_to_player( player.websocket,
                    json.dumps({
                        "type": "reconnected"
                    })
                )
                # Re-send history only to this reconnected player
                await manager.send_to_player( player.websocket,
                    json.dumps({
                        "type": "history",
                        "called_numbers": game.called_numbers
                    })
                )

                # DAVE: I keep getting cards[0] error, cards is null
                # Clean this issue up when refactoring to support multi-cards.
                if len(player.cards) == 0:
                    await manager.send_to_player( websocket,
                        json.dumps({
                            "type": "cards",
                            "cards": None
                        })
                    ) 
                else:
                    await manager.send_to_player( websocket,
                        json.dumps({
                            "type": "cards",
                            "cards": cards_to_dict(player.cards)
                        })
                    ) 
                # Debug
                logger.debug("Post-reconnect: Players:")
                for player in game.players.values():
                    print(
                        player.display_name,
                        player.connected,
                        player.websocket is not None
                    )

            elif data["type"] == "leave":
                player = game.get_player(data["player_id"])
                if player is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "reconnect_failed" # TODO: is this the right msg?
                        })
                    )
                else:
                    game.remove_player(player.player_id)
                    disconnect_player(websocket)
                await manager.send_to_player( websocket,
                    json.dumps({"type": "left",
                                "message": "Bye"})
                )
            
            # TODO: This goes away when we implement OCR scanning / card inputs.
            elif data["type"] == "load_test_cards":
                if game.status != GameStatus.SETUP:
                    logger.info("load_test_cards: Can only load cards during status SETUP")
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "reason": "Can only load cards during status SETUP"
                        })
                    )
                    continue
                
                player_id = data["player_id"]
                number = int(data["number"])
                player = game.get_player(player_id)
                if player is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "reconnect_failed"
                        })
                    )
                    continue
                logger.debug("In load_test_cards block, player "+ str(player)+
                      " will load grid "+ str(number));
                # player.dispose_cards() 
                player.add_card(create_test_card(player, number))
                await manager.send_to_player( websocket,
                    json.dumps({
                        "type": "cards",
                        "cards": cards_to_dict(player.cards)
                    })
                )     
                           
            elif data["type"] == "dispose_cards":
                logger.info("Entering disposeCards block")
                player_id = data["player_id"]
                player = game.get_player(player_id)
                if player is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "reconnect_failed"
                        })
                    )
                    continue
                logger.debug("disposeCards for player "+ str(player))
                # DAVE: null player when server restarts but browser doesn't.
                if player is not None:
                    player.dispose_cards()
                logger.debug("after disposeCards for player "+ str(player))
                await manager.send_to_player( websocket,
                    json.dumps({
                        "type": "cards",
                        "cards": []
                    })
                )   
                  
            elif data["type"] == "setup_new_game":
                logger.info("setup_new_game block")
                game.setup_new_game() # Also sets status to SETUP
                await manager.broadcast(
                    json.dumps({
                        "type": "new_game"
                    })
                )
                await manager.broadcast(
                    json.dumps({
                        "type": "game_status",
                        "status": str(game.status)
                    })
                )
                await broadcast_history(game)
                for player in game.players.values():
                    await send_cards_to_player(player)
                
            elif data["type"] == "start_game":
                logger.info("start_game: switch status to IN_PROGRESS")
                game.set_game_status(GameStatus.IN_PROGRESS)
                await manager.broadcast(
                    json.dumps({
                        "type": "game_status",
                        "status": str(game.status)
                    })
                )
            elif data["type"] == "request_game_status":
                logger.debug("request_game_status: sending status "+str(game.status)+" to client")
                await manager.send_to_player( websocket,
                    json.dumps({
                        "type": "game_status",
                        "status": str(game.status)
                    })
                )     

            elif data["type"] == "game_over":
                game.set_game_status(GameStatus.GAME_OVER)
                logger.info("game_over: broadcasting status=GAME_OVER change")
                await manager.broadcast(
                    json.dumps({
                        "type": "game_status",
                        "status": str(game.status)
                    })
                )

            elif data["type"] == "ping":
                logger.debug("ping: reply with pong")
                await manager.send_to_player( websocket,
                    json.dumps({
                        "type": "pong",
                    })
                )     

    except WebSocketDisconnect:
        disconnect_player(websocket)

        logger.info(f"Clients connected: {len(manager.active_connections)}")
        # Debug
        logger.debug("Post-disconnect: Players:")
        for player in game.players.values():
            print(
                player.display_name,
                player.connected,
                player.websocket is not None
            )

async def broadcast_history(game):
    logger.info("broadcast_history: entering with game="+ str(game))
    await manager.broadcast(
        json.dumps({
            "type": "history",
            "called_numbers": game.called_numbers
        })
    )

def cards_to_dict(cards):
    return [
        card.to_dict()
        for card in cards
    ]

async def send_cards_to_player(player: Player):
    logger.info("send_cards_to_player: player = "+str(player))
    if player.websocket is None:
        return
    if player.cards is not None and len(player.cards) > 0:
        await manager.send_to_player(
            player.websocket,
            json.dumps({
                "type": "cards",
                "cards": cards_to_dict(player.cards)
            })
        )     
    else:
        await manager.broadcast(
            json.dumps({
                "type": "cards",
                "cards": []
            })
        )     
            