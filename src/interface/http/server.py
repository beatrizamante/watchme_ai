import logging
import uvicorn

def make_server():
    logger = logging.getLogger("watchmeai")
    logger.info("Configuring server...")

    config = uvicorn.Config("main:app",
                        host="0.0.0.0",
                        port=5000,
                        log_level="debug",
                        use_colors=True
                        )
    server = uvicorn.Server(config)
    logger.info("Server configured successfully")
    return server
