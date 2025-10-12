from fastapi import File, HTTPException, UploadFile


def create_person_embedding(file: UploadFile = File(...)):
    """Embed a given user image
    Args: 
        file: person image for embedding
    Raises:
        ValueErrorException: if no boxes are found
        Exception: in case the AI cannot process the frame 
    Returns:
        embed: Embedded image if any are found
    """

    try:
        encoding = encode(file)
        if not encoding:
            raise ValueError("No person detected, please, try with another image")
        
        return encoding
    except Exception as e:
        raise e
   