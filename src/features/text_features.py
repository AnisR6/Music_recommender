"""Metadata text features (the original project's only content signal)."""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def metadata_text(df: pd.DataFrame) -> pd.Series:
    # artists as single tokens so "Daft Punk" is one feature, not "daft" + "punk"
    artists = df["artist_list"].map(lambda a: " ".join("artist_" + x.lower().replace(" ", "_") for x in a))
    return df["track_name"].str.lower() + " " + df["album_name"].str.lower() + " " + artists


class TextEmbedder:
    def __init__(self, dim=64, seed=42):
        self.tfidf = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True,
                                     token_pattern=r"(?u)\b[\w]{2,}\b")
        self.svd = TruncatedSVD(dim, random_state=seed)

    def fit_transform(self, df):
        T = self.tfidf.fit_transform(metadata_text(df))
        return normalize(self.svd.fit_transform(T))
