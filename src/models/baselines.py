import numpy as np
import scipy.sparse as sp


class PopularityRecommender:
    """Most-added tracks first: the playlist-cold-start fallback and the bar to beat."""
    name = "Popularity"

    def fit(self, train: sp.csr_matrix):
        self.counts = np.asarray(train.sum(0)).ravel().astype(np.float32)
        return self

    def score(self, seeds):
        return np.tile(np.log1p(self.counts), (seeds.shape[0], 1))


class RandomRecommender:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, train):
        self.n = train.shape[1]
        return self

    def score(self, seeds):
        return self.rng.random((seeds.shape[0], self.n), dtype=np.float32)
