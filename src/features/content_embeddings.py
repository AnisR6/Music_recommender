"""Combine audio, genre and text blocks into one content embedding per track."""
import numpy as np
from sklearn.preprocessing import normalize

DEFAULT_BLOCK_WEIGHTS = {"audio": 1.0, "genre": 0.8, "text": 0.35}


def combine(blocks: dict, weights: dict = None) -> np.ndarray:
    """Each block is L2-normalised per track, scaled by its weight, then concatenated.

    Normalising first stops a block with many dimensions from dominating just
    because it is wider. The final vector is normalised again so a dot product is
    a cosine similarity.
    """
    weights = weights or DEFAULT_BLOCK_WEIGHTS
    parts = [np.sqrt(weights[name]) * normalize(blocks[name]) for name in weights if weights[name] > 0]
    return normalize(np.hstack(parts)).astype(np.float32)
