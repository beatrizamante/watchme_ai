from fastapi import FastAPI
from src.interface.http.routes.person_embedding_handler import router as person_router
from src.interface.http.server import make_server
from src.interface.websocket.websocket_protocol import ws_router

app = FastAPI()

app.include_router(person_router)
app.include_router(ws_router)

if __name__ == "__main__":
    server, logger = make_server()
    server.run()
    logger.debug("Housten, we have a %s", 'thorny problem')
