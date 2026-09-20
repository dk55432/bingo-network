from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from connection_manager import ConnectionManager
from game import Game, GameManager, GameStatus, canonicalize_number
from game_store import GameStore
from player import Player
from bingo_card import BingoCard
from bingo_card_factory import create_test_card
import patterns_parser
from bingo_scan import router as scan_router, numeric_grid_to_labeled_grid, GRID_SIZE
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

manager = ConnectionManager()
game_manager = GameManager()

STATE_DB_PATH = os.getenv(
    "BINGO_STATE_DB",
    str(Path(__file__).resolve().parent / "data" / "bingo.sqlite3"),
)
store = GameStore(STATE_DB_PATH)

# Autosave cadence: game sets are small, so an unconditional whole-world
# snapshot every few seconds is cheap and needs no mutation tracking (no
# risk of missing a handler that changed state).
AUTOSAVE_INTERVAL_SECONDS = 5


def _flush_state():
    if game_manager.games:
        try:
            store.save_all(game_manager.snapshot())
        except Exception:
            logger.exception("_flush_state: failed to persist games")


async def _autosave_loop():
    while True:
        await asyncio.sleep(AUTOSAVE_INTERVAL_SECONDS)
        _flush_state()


@asynccontextmanager
async def lifespan(app):
    persisted = store.load_all()
    if persisted:
        game_manager.restore_from(persisted, WINNING_PATTERNS)
        logger.info(
            f"lifespan: restored {len(persisted)} game(s) from "
            f"{STATE_DB_PATH}"
        )
    task = asyncio.create_task(_autosave_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        _flush_state()
        store.close()


def _persist_game(game):
    try:
        store.upsert(game.game_id, game.to_persistable())
    except Exception:
        logger.exception(
            f"_persist_game: failed to persist game {game.game_id}"
        )

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(scan_router)
app.state.game_manager = game_manager
app.state.connection_manager = manager

templates = Jinja2Templates(directory="templates")

# Loaded once at server startup. Read-only after this point — every game
# shares the same registry of available winning patterns; a game only
# ever holds a reference to one entry (see Game.set_winning_pattern),
# never its own copy.
PATTERNS_CONFIG_PATH = "patterns_config.txt"
DEFAULT_PATTERN_NAME = "standard-bingo"

WINNING_PATTERNS = patterns_parser.load_patterns_from_file(PATTERNS_CONFIG_PATH)
if DEFAULT_PATTERN_NAME not in WINNING_PATTERNS:
    raise RuntimeError(
        f"{PATTERNS_CONFIG_PATH!r} must define a PATTERN named "
        f"{DEFAULT_PATTERN_NAME!r} — found: {list(WINNING_PATTERNS.keys())}"
    )
logger.info(f"Loaded {len(WINNING_PATTERNS)} winning pattern(s): {list(WINNING_PATTERNS.keys())}")

@app.get("/")
async def join_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="landing.html"
    )

@app.get("/join")
async def join_page(request: Request):
    resp = templates.TemplateResponse(
        request=request,
        name="join.html"
    )
    # The phone's join page JS changed — never let the browser serve a
    # stale cached copy, or old bugs (lost joins) survive a refresh.
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
    
@app.get("/player")
async def player_page(request: Request):
    resp = templates.TemplateResponse(
        request=request,
        name="player.html"
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/create_game")
async def create_game_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="create_game.html"
    )   

@app.get("/host")
async def host(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="host.html"
    )   

@app.get("/scan")
async def scan_page(request: Request):
    resp = templates.TemplateResponse(
        request=request,
        name="scan.html"
    )
    # The scan page's CSS changes often while the phone keeps a cached copy
    # (phones cache HTML hard), which bits of stale layout (broken border
    # "fragments") no restart seems to fix.  Never let the browser cache it.
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

def disconnect_player(websocket, game):
    if websocket in manager.active_connections:
        manager.disconnect(websocket)

    game.touch()

    if game.host_websocket == websocket:
        game.host_websocket = None
        logger.debug("host disconnected from game "+game.game_id)
        return

    for player in game.players.values():
        if player.websocket == websocket:
            player.connected = False
            player.websocket = None
            logger.debug(f"{player.display_name} disconnected")
            break

    for pending_id, entry in list(game.waiting_room.items()):
        if entry["websocket"] == websocket:
            del game.waiting_room[pending_id]
            logger.debug(f"waiting-room entry '{entry['display_name']}' disconnected")
            break
        
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info(f"Clients connected: {len(manager.active_connections)}")

    # Set once this connection is associated with a game, via "join",
    # "reconnect", "host_reconnect", or "create_game". Everything else
    # on this connection (submit_number, leave, disconnect cleanup, etc.)
    # looks the game up via game_manager.get_game(current_game_id).
    current_game_id = None

    try:
        # Client drives liveness with its own app-level {"type":"ping"}
        # every 10s (server replies pong), so the idle timeout must exceed
        # that cadence. There is no protocol-level WebSocket.ping() in
        # starlette — calling one here raised AttributeError, broke the
        # loop, and leaked every reconnecting socket from active_connections.
        # Set to 60s to tolerate background-tab throttling (browsers throttle
        # setInterval to ~1/min in inactive tabs).
        HEARTBEAT_INTERVAL = 60
        
        while True:
            try:
                # Wait for message with timeout for heartbeat
                message = await asyncio.wait_for(websocket.receive_text(), timeout=HEARTBEAT_INTERVAL)
                logger.info("Received: %s", message)
            except asyncio.TimeoutError:
                # Truly idle for HEARTBEAT_INTERVAL — a healthy client beats
                # this with its 10s app-level ping. If the link actually
                # died, the next receive_text() raises WebSocketDisconnect
                # and the finally block below unregisters the socket.
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.info(f"Invalid JSON received: {message!r}")
                continue

            if data["type"] == "ping":
                logger.info("ping received from client, sending pong")
                await manager.send_to_player(websocket,
                    json.dumps({
                        "type": "pong",
                    })
                )
                continue

            # This is only for Players.
            if data["type"] == "join":
                game_id = data.get("game_id")
                game = game_manager.get_game(game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "join",
                            "reason": "invalid_game_id"
                        })
                    )
                    continue

                current_game_id = game_id
                game.touch()

                if game.status != GameStatus.SETUP:
                    logger.info("join: game in progress, adding to waiting room")
                    pending_id = str(uuid.uuid4())
                    game.waiting_room[pending_id] = {
                        "display_name": data["display_name"],
                        "websocket": websocket
                    }
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "waiting_for_next_game",
                            "status": game.status.name,
                            "pattern_name": game.winning_pattern_name
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
    
                logger.debug("Post-join: Players:")
                for player in game.players.values():
                    logger.debug(
                        "%s connected=%s websocket=%s",
                        player.display_name,
                        player.connected,
                        player.websocket is not None,
                    )
                    
            # A message from Host
            elif data["type"] == "create_game":
                removed = game_manager.cleanup_empty_games()
                if removed:
                    logger.info(f"create_game: cleaned up {len(removed)} abandoned game(s)")
                game = game_manager.create_game()
                if game is None:
                    logger.error("create_game: create_game() returned None")
                    continue
                logger.info("Created game_id "+game.game_id)
                game.host_websocket = websocket
                current_game_id = game.game_id
                game.set_winning_pattern(DEFAULT_PATTERN_NAME, WINNING_PATTERNS[DEFAULT_PATTERN_NAME])
                # Crash-proof creation: the game must exist on disk before
                # we tell the host it exists.
                _persist_game(game)
                await manager.send_to_player(
                    websocket,
                    json.dumps({
                        "type": "game_created",
                        "game_id": game.game_id,
                        "available_patterns": list(WINNING_PATTERNS.keys())
                    })
                )
                await manager.send_to_player(
                    websocket,
                    json.dumps({
                        "type": "game_status",
                        "status": game.status.name,
                        "pattern_name": game.winning_pattern_name
                    })
                )
                
                    
            elif data["type"] == "submit_number":
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "submit_number",
                            "reason": "no_active_game"
                        })
                    )
                    continue
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
                # A called number is exactly what must survive a restart —
                # write it through immediately, don't wait for autosave.
                _persist_game(game)
                logger.debug("submit_number: broadcasting Hx: "+str(game.called_numbers))
                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "history",
                        "called_numbers": game.called_numbers
                    }),
                    game
                )
                
                winners = result["winners"]

                logger.debug("WINNERS: %s", winners)

                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "number_called",
                        "number": number
                    }),
                    game
                )

                if winners:
                    await manager.broadcast_to_game(
                        json.dumps({
                            "type": "winner",
                            "winners": winners
                        }),
                        game
                    )
                    logger.info("WINNER EVENT SENT")        
        
            elif data["type"] == "reconnect":
                # existing player reconnect logic, but scoped to game_manager.get_game(game_id)
                game_id = data.get("game_id")
                game = game_manager.get_game(game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({"type": "reconnect_failed"})
                    )
                    continue

                player_id = data.get("player_id")
                if not player_id:
                    await websocket.send_json({"type":"error", "message":"player_id required"})
                    continue

                current_game_id = game_id
                game.touch()
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

                # player.cards may be empty (e.g. after a server restart), so send
                # None rather than indexing [0] — the client clears its board.
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
                logger.debug("Post-reconnect: Players:")
                for player in game.players.values():
                    logger.debug(
                        "%s connected=%s websocket=%s",
                        player.display_name,
                        player.connected,
                        player.websocket is not None,
                    )

            elif data["type"] == "leave":
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({"type": "reconnect_failed"})
                    )
                    continue
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
                    disconnect_player(websocket, game)
                await manager.send_to_player( websocket,
                    json.dumps({"type": "left",
                                "message": "Bye"})
                )
            
            # TODO: This goes away when we implement OCR scanning / card inputs.
            elif data["type"] == "load_test_cards":
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({"type": "reconnect_failed"})
                    )
                    continue
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

            elif data["type"] == "add_scanned_card":
                # Client already ran /scan-card (REST — see bingo_scan.py)
                # and let the player review/correct the result. This adds
                # the final grid as a real card, same as load_test_cards
                # does for canned ones — no database involved, it just
                # goes straight into player.cards like every other card.
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({"type": "reconnect_failed"})
                    )
                    continue
                if game.status != GameStatus.SETUP:
                    logger.info("add_scanned_card: Can only add cards during status SETUP")
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "add_scanned_card",
                            "reason": "Can only add cards during status SETUP"
                        })
                    )
                    continue

                player_id = data.get("player_id")
                player = game.get_player(player_id) if player_id else None
                if player is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({"type": "reconnect_failed"})
                    )
                    continue

                numeric_grid = data.get("grid")
                try:
                    labeled_grid = numeric_grid_to_labeled_grid(numeric_grid)
                except (ValueError, TypeError) as e:
                    logger.info(f"add_scanned_card: rejected grid — {e}")
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "add_scanned_card",
                            "reason": str(e)
                        })
                    )
                    continue

                card = BingoCard(card_id=None, player_id=player.player_id, grid=labeled_grid)
                player.add_card(card)
                logger.info(f"add_scanned_card: added card {card.card_id} for player {player.display_name}")
                await manager.send_to_player( websocket,
                    json.dumps({
                        "type": "cards",
                        "cards": cards_to_dict(player.cards)
                    })
                )

            elif data["type"] == "dispose_cards":
                logger.info("Entering disposeCards block")
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({"type": "reconnect_failed"})
                    )
                    continue
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
                # player can be None if the server restarted while the
                # browser stayed open — dispose nothing and still ack.
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
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "setup_new_game",
                            "reason": "no_active_game"
                        })
                    )
                    continue
                game.setup_new_game() # Also sets status to SETUP
                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "new_game"
                    }),
                    game
                )
                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "game_status",
                        "status": game.status.name,
                        "pattern_name": game.winning_pattern_name
                    }),
                    game
                )
                await broadcast_history(game)
                for player in game.players.values():
                    await send_cards_to_player(player)

                # Promote anyone who tried to join mid-game into real
                # players now that we're back in SETUP.
                for pending_id, entry in list(game.waiting_room.items()):
                    ws = entry["websocket"]
                    del game.waiting_room[pending_id]
                    if ws not in manager.active_connections:
                        # They disconnected while waiting — nothing to do.
                        continue
                    new_player = Player(
                        player_id=str(uuid.uuid4()),
                        display_name=entry["display_name"],
                        websocket=ws,
                        connected_at=datetime.now()
                    )
                    game.add_player(new_player)
                    logger.info("setup_new_game: promoted waiting-room entry "+new_player.display_name)
                    await manager.send_to_player(
                        ws,
                        json.dumps({
                            "type": "joined",
                            "player_id": new_player.player_id
                        })
                    )
                    await manager.send_to_player(
                        ws,
                        json.dumps({
                            "type": "cards",
                            "cards": cards_to_dict(new_player.cards)
                        })
                    )
                    await manager.send_to_player(
                        ws,
                        json.dumps({
                            "type": "history",
                            "called_numbers": game.called_numbers
                        })
                    )
                
            elif data["type"] == "start_game":
                logger.info("start_game: switch status to IN_PROGRESS")
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "start_game",
                            "reason": "no_active_game"
                        })
                    )
                    continue
                game.set_game_status(GameStatus.IN_PROGRESS)
                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "game_status",
                        "status": game.status.name,
                        "pattern_name": game.winning_pattern_name
                    }),
                    game
                )
                await notify_waiting_room(game)
                
            elif data["type"] == "request_game_status":
                # join.html sends game_id explicitly since that connection
                # hasn't joined a game yet, so current_game_id isn't set.
                # Already-joined/reconnected clients omit it and fall back
                # to the game this connection is already attached to.
                lookup_id = data.get("game_id") or current_game_id
                game = game_manager.get_game(lookup_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "request_game_status",
                            "reason": "no_active_game"
                        })
                    )
                    continue
                logger.debug("request_game_status: sending status "+game.status.name+" to client")
                await manager.send_to_player( websocket,
                    json.dumps({
                        "type": "game_status",
                        "status": game.status.name,
                        "pattern_name": game.winning_pattern_name
                    })
                )     

            elif data["type"] == "game_over":
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "game_over",
                            "reason": "no_active_game"
                        })
                    )
                    continue
                game.set_game_status(GameStatus.GAME_OVER)
                logger.info("game_over: broadcasting status=GAME_OVER change")
                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "game_status",
                        "status": game.status.name,
                        "pattern_name": game.winning_pattern_name
                    }),
                    game
                )
                await notify_waiting_room(game)

            elif data["type"] == "set_winning_pattern":
                game = game_manager.get_game(current_game_id)
                if game is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "set_winning_pattern",
                            "reason": "no_active_game"
                        })
                    )
                    continue
                if game.status != GameStatus.SETUP:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "set_winning_pattern",
                            "reason": "can_only_change_during_setup"
                        })
                    )
                    continue
                pattern_name = data.get("pattern_name")
                pattern = WINNING_PATTERNS.get(pattern_name)
                if pattern is None:
                    await manager.send_to_player(
                        websocket,
                        json.dumps({
                            "type": "action_rejected",
                            "action": "set_winning_pattern",
                            "reason": "unknown_pattern"
                        })
                    )
                    continue
                game.set_winning_pattern(pattern_name, pattern)
                logger.info(f"set_winning_pattern: game {game.game_id} now using {pattern_name!r}")
                await manager.broadcast_to_game(
                    json.dumps({
                        "type": "game_status",
                        "status": game.status.name,
                        "pattern_name": game.winning_pattern_name
                    }),
                    game
                )

            elif data["type"] == "host_reconnect":
                game_id = data.get("game_id")
                game = game_manager.get_game(game_id)
                if not game:
                    await websocket.send_json({"type": "host_reconnect_failed"})
                    continue
                game.host_websocket = websocket
                current_game_id = game_id
                game.touch()
                await websocket.send_json({
                    "type": "host_reconnected",
                    "available_patterns": list(WINNING_PATTERNS.keys())
                })
                await websocket.send_json({
                    "type": "game_status",
                    "status": game.status.name,
                    "pattern_name": game.winning_pattern_name
                })
                await websocket.send_json({
                    "type": "history",
                    "called_numbers": game.called_numbers
                })    

            # Catch-all
            else:
                logger.error("Unexpected message: "+data["type"])
                continue

    except WebSocketDisconnect:
        game = game_manager.get_game(current_game_id)
        if game is not None:
            # existing disconnect cleanup, scoped to the right game
            disconnect_player(websocket, game)

            logger.info(f"Clients connected: {len(manager.active_connections)}")
            logger.debug("Post-disconnect: Players:")
            for player in game.players.values():
                logger.debug(
                    "%s connected=%s websocket=%s",
                    player.display_name,
                    player.connected,
                    player.websocket is not None,
                )
        else:
            # A host connection that never reached host_reconnect/create_game,
            # or a browser tab closed before joining a game.
            if websocket in manager.active_connections:
                manager.disconnect(websocket)
            logger.info(f"Clients connected: {len(manager.active_connections)}")

    finally:
        # Every exit path (WebSocketDisconnect, broken pipe, or an unexpected
        # exception during receive/send) must unregister this socket, or the
        # client count ratchets up one per reconnect with no decrement ever.
        # This runs for ALL exits — not just WebSocketDisconnect — and it is
        # idempotent (disconnect_player/disconnect no-op once already removed).
        game = game_manager.get_game(current_game_id)
        if game is not None:
            disconnect_player(websocket, game)
        elif websocket in manager.active_connections:
            manager.disconnect(websocket)

async def notify_waiting_room(game):
    """Let people queued in the waiting room see status changes (e.g. the
    game moving to GAME_OVER) — a small courtesy so the wait doesn't feel
    like a black box, even though they aren't promoted until SETUP."""
    for entry in game.waiting_room.values():
        await manager.send_to_player(
            entry["websocket"],
            json.dumps({
                "type": "game_status",
                "status": game.status.name,
                "pattern_name": game.winning_pattern_name
            })
        )

async def broadcast_history(game):
    logger.info("broadcast_history: entering with game="+ str(game))
    await manager.broadcast_to_game(
        json.dumps({
            "type": "history",
            "called_numbers": game.called_numbers
        }),
        game
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
        # Was previously manager.broadcast(...) — that sent an empty
        # cards list to EVERY connection, wiping other players' boards.
        # This should only ever go to the one player being updated.
        await manager.send_to_player(
            player.websocket,
            json.dumps({
                "type": "cards",
                "cards": []
            })
        )     
            