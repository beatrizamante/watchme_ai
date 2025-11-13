import logging
import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException
from src.application.use_cases.create_person_embedding import create_person_embedding
from src.application.use_cases.predict_person import predict_person_on_stream
from src.domain.DetectionModel import FindPersonRequest
from src.domain.Image import ImageModel

logger = logging.getLogger("watchmeai")
router = APIRouter()

@router.post("/upload-embedding")
async def upload_person_image(request: ImageModel):
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
            "status": "success"
        }


    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Failed to process image: {str(e)}")

@router.post("/find")
async def predict_person(request: FindPersonRequest):
    """Search requisition for person of interest in a video or stream"""
    logger.info("Starting person search")
    logger.debug(f"Request: {request.video} and {request.person}")


    #try:
        #matches = predict_person_on_stream(person.embedding, video.path)
        #logger.info(f"Found {len(matches)} matches")
        #logger.debug(f"Matches: {matches}")
        #return {"matches": matches}

    #except Exception as e:
        #logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        #raise HTTPException(status_code=500, detail=str(e))
