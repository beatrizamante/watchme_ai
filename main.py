from fastapi import FastAPI
from src.interface.http.handlers.person_embedding_handler import router as person_router
from src.interface.http.handlers.video_feed_handler import router as video_router
from src.interface.websocket.websocket_protocol import ws_router


app = FastAPI()

app.include_router(person_router)
app.include_router(video_router)
app.include_router(ws_router)
