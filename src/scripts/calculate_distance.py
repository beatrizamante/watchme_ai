from typing import List

import numpy as np


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


def compute_similarity(features1, features2):
    """
    Compute cosine similarity between two feature vectors or matrices.

    Args:
        features1: Feature vector/matrix (1D or 2D numpy array)
        features2: Feature vector/matrix (1D or 2D numpy array)

    Returns:
        float or numpy.ndarray: Similarity score(s)
    """
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)

    similarity = np.dot(features1, features2.T) / (
        np.linalg.norm(features1, axis=1, keepdims=True) *
        np.linalg.norm(features2, axis=1, keepdims=True).T
    )

    if similarity.shape == (1, 1):
        return similarity[0, 0]

    return similarity


def find_most_similar(query_features, gallery_features, top_k=5):
    """
    Find most similar images in gallery for a query.

    Args:
        query_features: Query feature vector (1D numpy array)
        gallery_features: Gallery feature matrix (2D numpy array)
        top_k: Number of top similar images to return

    Returns:
        tuple: (indices, similarities) of top_k most similar images
    """
    similarities = compute_similarity(query_features, gallery_features)

    if similarities.ndim == 2:
        similarities = similarities.flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_similarities = similarities[top_indices]

    return top_indices, top_similarities
