import logging
import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, Depends

from src._lib.decrypt import DecryptionService
from src.application.use_cases.create_person_embedding import create_person_embedding
from src.application.use_cases.predict_person import predict_person_on_stream
from src.domain.Detection import FindPersonRequest
from src.domain.Image import Image
from src._lib.container import get_container
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src._lib.encrypt import EncryptionService

logger = logging.getLogger("watchmeai")
router = APIRouter()

def get_osnet_encoder() -> OSNetEncoder:
    """
    Retrieves an instance of OSNetEncoder from the application's dependency injection container.

    Returns:
        OSNetEncoder: An encoder object for extracting person embeddings using the OSNet model.
    """
    return get_container().osnet_encoder()

def get_encryption_service() -> EncryptionService:
    """
    Retrieves an instance of the EncryptionService from the application's dependency injection container.

    Returns:
        EncryptionService: An instance of the encryption service used for cryptographic operations.
    """
    return get_container().encryption_service()

def get_decryption_service() -> DecryptionService:
    """
    Retrieves an instance of the EncryptionService from the application's dependency injection container.

    Returns:
        EncryptionService: An instance of the encryption service used for cryptographic operations.
    """
    return get_container().decryption_service()


@router.post("/upload-embedding")
async def upload_person_image(
    request: Image,
    encoder: OSNetEncoder = Depends(get_osnet_encoder),
    encryption_service: EncryptionService = Depends(get_encryption_service)
):
    """
    Upload an image and get the person embedding.
    """

    try:
        image_bytes = base64.b64decode(request.image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, 1)

        if image is None:
            raise ValueError("Could not decode image file")

        embedding = create_person_embedding(image, encoder, encryption_service)

        return {
            "embedding": embedding,
            "status": "success",
            "method": "multi_frame"
        }
    except Exception as e:
        logging.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=422, detail={"Failed to process image: %s", str(e)}) from e

@router.post("/find")
async def predict_person(request: FindPersonRequest, encoder: OSNetEncoder = Depends(get_osnet_encoder), decryption_service: DecryptionService = Depends(get_decryption_service) ):
    """Search requisition for person of interest in a video or stream"""
    logger.info("Starting person search")

    try:
        matches = predict_person_on_stream(request.person.embedding, request.video.path, decryption_service, encoder)
        logger.info("Found %s matches", len(matches))
        logger.debug("Matches: %s", matches)
        return {"matches": matches}

    except Exception as e:
        logger.error({"Error during prediction: %s", str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
