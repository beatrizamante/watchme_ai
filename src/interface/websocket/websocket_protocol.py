import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.interface.http.handlers.stream_handler import handle_single_frame, handle_start_tracking, handle_stop_tracking
from src.interface.http.handlers.websocket_handler import ConnectionHandler

manager = ConnectionHandler()
ws_router = APIRouter()

@ws_router.websocket("/video-stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "start_tracking":
                await handle_start_tracking(session_id, data, manager)

            elif message_type == "single_frame":
                await handle_single_frame(session_id, data, manager)

            elif message_type == "stop_tracking":
                await handle_stop_tracking(session_id, manager)

            elif message_type == "ping":
                await manager.send_status(session_id, "pong", "Connection active")

            else:
                await manager.send_error(session_id, f"Unknown message type: {message_type}")

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logging.error(f"WebSocket error for {session_id}: {e}")
        await manager.send_error(session_id, f"WebSocket error: {str(e)}")
    finally:
        manager.disconnect(session_id)
