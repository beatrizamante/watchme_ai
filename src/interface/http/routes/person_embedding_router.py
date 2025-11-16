import logging
import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException
from src.application.use_cases.create_person_embedding import create_person_embedding
from src.application.use_cases.predict_person import predict_person_on_stream
from src.domain.Detection import FindPersonRequest
from src.domain.Image import Image

logger = logging.getLogger("watchmeai")
router = APIRouter()

@router.post("/upload-embedding")
async def upload_person_image(request: Image):
    """
    Upload an image and get the person embedding.
    """
    try:
        image_bytes = base64.b64decode(request.image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image file")

        embedding = create_person_embedding(image)

        return {
            "embedding": embedding,
            "status": "success",
            "method": "multi_frame"
        }
    except Exception as e:
        logging.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=422, detail=f"Failed to process image: {str(e)}") from e

@router.post("/find")
async def predict_person(request: FindPersonRequest):
    """Search requisition for person of interest in a video or stream"""
    logger.info("Starting person search")

    try:
        matches = predict_person_on_stream(request.person.embedding, request.video.path)
        logger.info(f"Found {len(matches)} matches")
        logger.debug(f"Matches: {matches}")
        return {"matches": matches}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
