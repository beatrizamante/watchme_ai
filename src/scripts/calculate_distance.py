import numpy as np
from typing import List

def calculate_distance(chosen_person: List[float], detected_person: List[float]):
    """This script calculates the cosine distance between embeddings
    Args:
        chosen_person: The person that we are trying to find in video;
        detected_person:The person found in the video.
    Returns:
        distance: The multiplied distance from the chosen_person with the detected_person;
    """
    chosen_embed = np.array(chosen_person)
    detected_embed = np.array(detected_person)
    
    distance = np.linalg.norm(chosen_embed - detected_embed)
    
    return distance

