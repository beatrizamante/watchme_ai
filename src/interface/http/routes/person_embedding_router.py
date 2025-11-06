import logging
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException
from src.application.use_cases.create_person_embedding import create_person_embedding
from src.application.use_cases.predict_person import predict_person_on_stream
from src.domain.Image import ImageModel
from src.domain.Person import Person
from src.domain.Video import Video

router = APIRouter()

@router.post("/upload-embedding")
async def upload_person_image(request: ImageModel):
    """
    Upload an image and get the person embedding.
    """
    print(f"Received base64 image")

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
async def predict_person(person: Person, video: Video):
    """
    Search requisition for person of interest in a video or stream
    """
    matches = predict_person_on_stream(person.embed, video.path)
    return { "matches": matches }
