import numpy as np
from sklearn.preprocessing import normalize


class ContentRecommender:
    """Ranks tracks by cosine similarity to the playlist's average content vector.

    Needs no interaction history at all, so it can rank tracks nobody has added yet.
    """
    name = "Content"

    def __init__(self, embeddings: np.ndarray, name=None):
        self.E = normalize(embeddings).astype(np.float32)
        if name: self.name = name

    def fit(self, train=None):
        return self

    def profile(self, seeds):
        return normalize(np.asarray(seeds @ self.E))

    def score(self, seeds):
        return self.profile(seeds) @ self.E.T

    def similar(self, item, k=10):
        s = self.E @ self.E[item]
        s[item] = -np.inf
        top = np.argsort(-s)[:k]
        return top, s[top]
