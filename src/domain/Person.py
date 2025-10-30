from typing import List
from typing_extensions import TypeAlias

from pydantic import BaseModel

class PersonModel(BaseModel):
    """Class model/domain for an embedded person"""
    name: str
    user_id: int
    embed: List[float]

Person: TypeAlias = PersonModel
