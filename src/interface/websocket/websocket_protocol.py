from fastapi import APIRouter, WebSocket

from src.application.use_cases.predict_person import predict_person_on_stream

ws_router = APIRouter()

@ws_router.websocket("/video-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        frame = data["frame"]
        person_embed = data["person_embed"]

        matches = predict_person_on_stream(person_embed, frame)
        await websocket.send_json({"matches": matches})
