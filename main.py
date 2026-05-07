from fastapi import FastAPI
from src.config.logging_config import setup_logging

from src.interface.http.server import make_server
from src.interface.http.routes.person_embedding_router import router as person_router
from src.interface.websocket.websocket_protocol import ws_router
from src._lib.container import get_container

logger = setup_logging()
container = get_container()

app = FastAPI()

app.include_router(person_router)
app.include_router(ws_router)

def run():
    """Entry point for uvx / uv tool run."""
    logger.info("Starting WatchMe AI Backend...")
    server = make_server()
    server.run()


if __name__ == "__main__":
    run()
