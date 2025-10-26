from typing import List

from src._lib.decrypt import decrypt_embedding
from src.scripts.calculate_distance import calculate_distance

def predict_person_on_stream(chosen_person: List[float], stream):
    """This method compares embeddings
    Args:
        chosen_person: The person that we are trying to find in video;
        people_list: A tuple of people embeddings and coordinates present on the video.
    Returns:
        is_match: True if the person is found or false if not;
        coordinate: The coordinates of the found person on the frames;
    """

    decrypted_embedding = decrypt_embedding(chosen_person, shape=(512,), dtype='float32')
    for person in people_list:
        distance = calculate_distance(decrypted_embedding, person)
        if distance >= 0.8:
            return True, person.coordinate

    return False, None
