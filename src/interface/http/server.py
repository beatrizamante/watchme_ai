import logging

import uvicorn


def make_server():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
    logger = logging.getLogger("uvicorn")

    config = uvicorn.Config("main:app",
                            host="0.0.0.0",
                            port=5000,
                            )
    server = uvicorn.Server(config)
    return server, logger
