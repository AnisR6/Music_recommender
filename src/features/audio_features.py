"""Feature engineering for Spotify audio features."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src import config

SKEWED = ["speechiness", "acousticness", "instrumentalness", "liveness"]
BOUNDED = ["danceability", "energy", "valence"]


def fold_tempo(bpm: pd.Series, low=70.0) -> pd.Series:
    """Fold tempo into one octave [low, 2*low).

    Beat trackers often report half or double the perceived tempo (an 87 BPM
    track read as 174). Folding makes 87 and 174 the same "feel", which matters
    for similarity more than the raw number.
    """
    t = bpm.astype(float).copy()
    while (t >= 2 * low).any():
        t = t.where(t < 2 * low, t / 2)
    while (t < low).any():
        t = t.where(t >= low, t * 2)
    return t


def circle_of_fifths(key: pd.Series, mode: pd.Series):
    """Map a key to an angle on the circle of fifths, minor keys at their relative major.

    A minor and C major share all their notes, so they sit at the same angle;
    neighbouring angles mix well (the idea behind DJ "Camelot" harmonic mixing).
    """
    pitch = np.where(mode == 1, key, (key + 3) % 12)
    pos = (pitch * 7) % 12
    ang = 2 * np.pi * pos / 12
    ang = np.where(key < 0, np.nan, ang)
    return np.sin(ang), np.cos(ang)


class AudioFeatureEngineer:
    """Turns raw audio columns into a model-ready, comparable feature block."""

    def __init__(self):
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                      random_state=config.SEED)
        self.scaler = StandardScaler()
        self.columns_ = None

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in BOUNDED:
            out[c] = df[c]
        out["loudness"] = df["loudness"].clip(lower=-35)
        tempo = fold_tempo(df["tempo"])
        out["tempo_feel"] = np.log2(tempo)                # perceived tempo on a log scale
        out["groove"] = df["danceability"] * df["energy"]  # interaction: danceable *and* driving
        out["log_duration"] = np.log(df["duration_ms"] / 1000)
        sin, cos = circle_of_fifths(df["key"].values, df["mode"].values)
        out["key_sin"], out["key_cos"] = np.nan_to_num(sin), np.nan_to_num(cos)
        out["major"] = df["mode"].astype(float)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._frame(df)
        skew = pd.DataFrame(self.qt.fit_transform(df[SKEWED]), columns=[f"q_{c}" for c in SKEWED], index=df.index)
        X = pd.concat([base, skew], axis=1)
        self.columns_ = list(X.columns)
        return pd.DataFrame(self.scaler.fit_transform(X), columns=self.columns_, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._frame(df)
        skew = pd.DataFrame(self.qt.transform(df[SKEWED]), columns=[f"q_{c}" for c in SKEWED], index=df.index)
        X = pd.concat([base, skew], axis=1)[self.columns_]
        return pd.DataFrame(self.scaler.transform(X), columns=self.columns_, index=df.index)


# How much each engineered column counts in the similarity. Key and mode are real
# but weak signals; energy/acousticness/valence drive most of how a track "feels".
FEATURE_WEIGHTS = {
    "danceability": 1.0, "energy": 1.2, "valence": 1.0, "loudness": 0.6, "tempo_feel": 0.7,
    "groove": 0.6, "log_duration": 0.3, "key_sin": 0.25, "key_cos": 0.25, "major": 0.3,
    "q_speechiness": 0.7, "q_acousticness": 1.1, "q_instrumentalness": 0.9, "q_liveness": 0.3,
}


def weighted(X: pd.DataFrame, weights=FEATURE_WEIGHTS) -> np.ndarray:
    return X.values * np.array([weights.get(c, 1.0) for c in X.columns])
