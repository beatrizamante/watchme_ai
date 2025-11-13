from pydantic import BaseModel
from src.domain.Person import Person
from src.domain.Video import Video

class FindPersonRequest(BaseModel):
    person: Person
    video: Video
