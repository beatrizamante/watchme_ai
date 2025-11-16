from fastapi import FastAPI
from src.config.logging_config import setup_logging

from src.interface.http.routes.person_embedding_router import router as person_router
from src.interface.http.routes.test_route import router as test_router

from src.interface.http.server import make_server
from src.interface.websocket.websocket_protocol import ws_router

logger = setup_logging()

app = FastAPI()

app.include_router(person_router)
app.include_router(test_router)
app.include_router(ws_router)

if __name__ == "__main__":
    logger.info("Starting WatchMe AI Backend...")
    server = make_server()
    server.run()
