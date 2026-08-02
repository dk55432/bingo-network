from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import asyncio
import json

connected_clients = []
app = FastAPI()

templates = Jinja2Templates(directory="templates")

def canonicalize_number(value):

    value = value.strip().upper()

    if value.isdigit():

        number = int(value)

        if 46 <= number <= 60:
            return f"G{number}"

    return value


class ConnectionManager:

    def __init__(self):
        self.active_connections = []


    async def connect(self, websocket: WebSocket):

        await websocket.accept()

        self.active_connections.append(websocket)


    def disconnect(self, websocket: WebSocket):

        self.active_connections.remove(websocket)


    async def broadcast(self, message: str):

        for connection in self.active_connections:

            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def player(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="player.html"
    )
    
async def broadcast(message: str):

    for client in connected_clients:
        await client.send_text(message)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await manager.connect(websocket)

    print(f"Clients connected: {len(manager.active_connections)}")

    try:

        # while True:

            # message = await websocket.receive_text()
            # data = json.loads(message)
            # print(data)
            # print(f"Received: {message}")
            # await manager.broadcast(message)
        while True:

            message = await websocket.receive_text()

            try:
                data = json.loads(message)

            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message!r}")
                continue
                
            if data["type"] == "submit_number":

                number = canonicalize_number(data["value"])

                event = {
                    "type": "number_called",
                    "number": number
                }

                await manager.broadcast(
                    json.dumps(event)
                )            

    except WebSocketDisconnect:

        manager.disconnect(websocket)

        print(f"Clients connected: {len(manager.active_connections)}")
    
        
@app.get("/host")
async def host(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="host.html"
    )