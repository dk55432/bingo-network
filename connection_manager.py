from fastapi import WebSocket

class ConnectionManager:

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        dead_connections = []
        # Debug
        print("Broadcasting to:")
        for i, ws in enumerate(self.active_connections):
            print(i, ws)
            
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except RuntimeError:
                dead_connections.append(connection)
        for connection in dead_connections:
            self.disconnect(connection)
                    
    async def send_to_player(self, websocket, message):
        if websocket is None:
            return
        await websocket.send_text(message)        
    