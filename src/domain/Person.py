from typing import List, Optional
from typing_extensions import TypeAlias

from pydantic import BaseModel

class PersonModel(BaseModel):
    """Class model/domain for an embedded person"""
    id: Optional[int] = None
    name: str
    embedding: str
    user_id: Optional[int] = None

Person: TypeAlias = PersonModel
