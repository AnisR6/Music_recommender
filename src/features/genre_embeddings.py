"""Dense genre vectors learned from how genres share tracks and how they sound."""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, normalize


class GenreEmbedder:
    def __init__(self, dim=24, co_weight=0.5, seed=42):
        self.dim, self.co_weight, self.seed = dim, co_weight, seed

    def fit(self, genres: pd.Series, audio: np.ndarray):
        self.mlb = MultiLabelBinarizer()
        G = self.mlb.fit_transform(genres).astype(float)          # tracks x genres
        self.genres_ = list(self.mlb.classes_)

        # 1) co-listing: PPMI of genres that label the same track
        co = G.T @ G
        total = co.sum()
        pg = co.sum(1, keepdims=True) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((co / total) / (pg @ pg.T))
        ppmi = np.nan_to_num(np.maximum(pmi, 0), posinf=0)
        np.fill_diagonal(ppmi, ppmi.max())

        # 2) sound: cosine between genre centroids in engineered audio space
        cent = normalize((G.T @ audio) / G.sum(0)[:, None])
        sound = (cent @ cent.T + 1) / 2

        sim = self.co_weight * ppmi / ppmi.max() + (1 - self.co_weight) * sound ** 3
        self.similarity_ = pd.DataFrame(sim, index=self.genres_, columns=self.genres_)
        svd = TruncatedSVD(self.dim, random_state=self.seed)
        self.vectors_ = normalize(svd.fit_transform(sim))
        self.G_ = G
        return self

    def track_vectors(self, genres: pd.Series = None) -> np.ndarray:
        G = self.G_ if genres is None else self.mlb.transform(genres).astype(float)
        return normalize(G @ self.vectors_)

    def nearest(self, genre, k=6):
        s = self.similarity_[genre].drop(genre).sort_values(ascending=False)
        return s.head(k)
