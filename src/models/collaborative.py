"""Collaborative filtering: item-kNN, ALS and BPR on the playlist x track matrix.

The original project used LightFM, which is no longer maintained and does not
build on current Python (3.12+). These use the maintained `implicit` library and a
small item-kNN, and they can all fold in a brand-new playlist from a few seed
tracks, which is what the console app needs.
"""
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.preprocessing import normalize

from src import config


class ItemKNN:
    """Item-item cosine similarity with shrinkage, keeping the top-N neighbours per item."""
    name = "Item-kNN"

    def __init__(self, k=config.KNN_NEIGHBOURS, shrink=10.0):
        self.k, self.shrink = k, shrink

    def fit(self, train: sp.csr_matrix):
        X = train.T.tocsr().astype(np.float32)                 # items x users
        norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        co = (X @ X.T).tocsr()
        co.setdiag(0); co.eliminate_zeros()
        r, c = co.nonzero()
        co.data = co.data / (norms[r] * norms[c] + self.shrink)
        # keep top-k per row
        rows, cols, vals = [], [], []
        for i in range(co.shape[0]):
            s, e = co.indptr[i], co.indptr[i + 1]
            if e == s: continue
            d, idx = co.data[s:e], co.indices[s:e]
            if len(d) > self.k:
                keep = np.argpartition(-d, self.k)[:self.k]
                d, idx = d[keep], idx[keep]
            rows.extend([i] * len(d)); cols.extend(idx); vals.extend(d)
        self.sim = sp.csr_matrix((vals, (rows, cols)), shape=co.shape, dtype=np.float32)
        return self

    def score(self, seeds):
        return (seeds @ self.sim).toarray()


class _ImplicitMF:
    """Shared fold-in for factor models: solve a new playlist's vector from its seed tracks."""

    def fit(self, train):
        self.model.fit(train.tocsr(), show_progress=False)
        self.Y = np.asarray(self.model.item_factors, dtype=np.float32)
        return self

    def user_vectors(self, seeds):
        Y, reg = self.Y, self.fold_reg
        YtY = Y.T @ Y
        out = np.zeros((seeds.shape[0], Y.shape[1]), np.float32)
        for r in range(seeds.shape[0]):
            items = seeds.indices[seeds.indptr[r]:seeds.indptr[r + 1]]
            if len(items) == 0: continue
            Ys = Y[items]
            A = YtY + self.alpha * Ys.T @ Ys + reg * np.eye(Y.shape[1])
            b = (1 + self.alpha) * Ys.sum(0)
            out[r] = np.linalg.solve(A, b)
        return out

    def score(self, seeds):
        return self.user_vectors(seeds) @ self.Y.T


class ALSRecommender(_ImplicitMF):
    name = "ALS"

    def __init__(self, factors=config.ALS_FACTORS, reg=config.ALS_REG, iterations=config.ALS_ITERS,
                 alpha=config.ALS_ALPHA, seed=config.SEED):
        self.alpha, self.fold_reg = alpha, reg * 10
        self.model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iterations,
                                             alpha=alpha, random_state=seed)


class BPRRecommender(_ImplicitMF):
    """Bayesian Personalised Ranking: learns from (added, not added) pairs.

    BPR factors are not least-squares factors, so the ALS fold-in does not apply.
    A new playlist is represented by the mean of its seed tracks' factors instead,
    which scored as well as BPR's own trained playlist vectors in notebook 05.
    """
    name = "BPR"

    def __init__(self, factors=256, lr=0.1, reg=0.001, iterations=300, seed=config.SEED):
        self.model = BayesianPersonalizedRanking(factors=factors, learning_rate=lr, regularization=reg,
                                                 iterations=iterations, random_state=seed)

    def fit(self, train):
        super().fit(train)
        full = self.Y
        self.Y, self.bias = full[:, :-1], full[:, -1]   # implicit appends an item-bias column
        return self

    def user_vectors(self, seeds):
        return np.asarray(normalize(seeds) @ self.Y)

    def score(self, seeds):
        return self.user_vectors(seeds) @ self.Y.T + self.bias
