from typing import TypeAlias
from pydantic import BaseModel
from typing import List
import numpy as np
class PersonModel(BaseModel):
    """Class model/domain for an embedded person"""
    name: str
    user_id: int
    embed: List[float]

Person: TypeAlias = PersonModel
