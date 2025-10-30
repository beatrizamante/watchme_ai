from typing_extensions import TypeAlias
from pydantic import BaseModel

class VideoModel(BaseModel):
    id: int
    user_id: int
    path: str

Video: TypeAlias = VideoModel
