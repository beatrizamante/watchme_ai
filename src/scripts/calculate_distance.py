from typing import List

def calculate_distance(chosen_person: List[float], detected_person: List[float]) -> float:
    """This script calculates the cosine distance between embeddings
    Args:
        chosen_person: The person that we are trying to find in video;
        detected_person:The person found in the video.
    Returns:
        distance: The multiplied distance from the chosen_person with the detected_person;
    """
    
    