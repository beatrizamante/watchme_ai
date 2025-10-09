 from fastapi import File, HTTPException, UploadFile
 
 def create_person_embedding(
     file: UploadFile = File(...)
 ):
     """Embed a given user image
     Args: 
         file: person image for embedding
        Raises:
        
     Returns:
         embed: Embedded image if any are found
     """
         
 
     try:
         encoding = encode(file)
         if not encoding:
             raise HTTPException(status_code=400, detail="No person detected, please, try with another image")
         
         return encoding
     except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
   