from typing_extensions import TypeAlias
from pydantic import BaseModel

from src.domain.Person import Person


class VideoRequest(BaseModel):
    person: Person
    video_path: str

Video: TypeAlias = VideoRequest
