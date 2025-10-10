import uvicorn
import logging

def create_server():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
    logger = logging.getLogger("uvicorn")

    config = uvicorn.Config("main:app", 
                            host="0.0.0.0", 
                            port=443, 
                            )
    server = uvicorn.Server(config)
    return server, logger