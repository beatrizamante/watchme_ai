from http.client import HTTPException

from src.domain.Person import PersonModel


def get_person_embedding(person: PersonModel):
    """Get a chosen person's embedding from the database

    Args:
        person: PersonModel

    Raises:
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        person
    """
     try:
         user_data =  {key: (value if key not in ["hash", "hash1"] else None) for key, value in user.__dict__.items()}        
         return user_data
     except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
 