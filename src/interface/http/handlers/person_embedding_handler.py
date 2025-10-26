from fastapi import APIRouter, UploadFile, File

from src.application.use_cases.create_person_embedding import create_person_embedding
from src.application.use_cases.predict_person import predict_person_on_stream
from src.domain.Person import Person

router = APIRouter()

@router.post("/upload-embedding")
async def upload_person_image(file: UploadFile = File(...)):
    """
    Upload an image and get the person embedding.
    """
    return create_person_embedding(file)

@router.post("/find/{personId}")
async def predict_person(person: Person, file: UploadFile = File(...)):
    """
    Search requisition for person of interest in a video or stream
    """
    matches = predict_person_on_stream(person.embed, file)
    return { "matches": matches }
