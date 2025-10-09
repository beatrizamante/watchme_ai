from typing import List

from src.scripts.calculate_distance import calculate_distance


def compare_embeddings(chosen_person: List[float], people_list: tuple):
    """This method compares embeddings
    Args:
        chosen_person: The person that we are trying to find in video;
        people_list: A tuple of people embeddings and coordinates present on the video.
    Returns:
        is_match: True if the person is found or false if not;
        coordinate: The coordinates of the found person on the frames;
    """
    
    for person in people_list:
        distance = calculate_distance(chosen_person, person)
        if(distance >= 0.8):
            return
        
    