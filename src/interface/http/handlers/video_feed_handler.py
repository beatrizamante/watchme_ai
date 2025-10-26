from fastapi import APIRouter

router = APIRouter()

@router.post("/api/upload-embedding")
async def upload_video(file):
    """
    Upload an image and get the person embedding.
    """
    return ""
