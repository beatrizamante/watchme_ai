from typing import Optional
from typing_extensions import TypeAlias
from pydantic import BaseModel

class VideoModel(BaseModel):
    id: int
    path: str
    user_id: Optional[int] = None
    created_at: Optional[str] = None
    username: Optional[str] = None

Video: TypeAlias = VideoModel
