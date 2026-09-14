"""Speaker Voice Verification module.

Extracts and compares acoustic voiceprint descriptors using cosine similarity
to distinguish the enrolled user from nearby bystanders and background chatter.
"""

import math
import logging
from typing import Optional

logger = logging.getLogger("voice_verifier")

DEFAULT_SIMILARITY_THRESHOLD = 0.70


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two acoustic embedding vectors."""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        min_len = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:min_len]
        vec_b = vec_b[:min_len]

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def verify_speaker_voice(
    candidate_descriptor: Optional[list[float]],
    enrolled_descriptor: Optional[list[float]],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[bool, float]:
    """Verify if candidate utterance voice matches the enrolled user profile.
    
    Returns (is_verified, similarity_score).
    If no enrolled descriptor exists yet, returns (True, 1.0) to allow baseline operation.
    """
    if not enrolled_descriptor:
        return True, 1.0

    if not candidate_descriptor:
        # If client did not supply candidate descriptor, allow fallback
        return True, 1.0

    similarity = cosine_similarity(candidate_descriptor, enrolled_descriptor)
    is_match = similarity >= threshold

    logger.info(
        "Speaker verification: similarity=%.4f, threshold=%.2f, match=%s",
        similarity,
        threshold,
        is_match,
    )
    return is_match, similarity
