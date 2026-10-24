"""One object that answers the questions the console app asks."""
import re
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import artifacts, config
from src.data.load import load_catalog
from src.models.hybrid import zscore

MOOD_DIMS = ["energy", "valence", "danceability", "acousticness", "tempo"]


def _fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


class MusicRecommender:
    def __init__(self):
        self.tracks = load_catalog()
        feats = artifacts.load_features()
        self.E = feats["embeddings"]
        self.audio = pd.DataFrame(feats["audio"], columns=feats["audio_cols"])
        self.genre_sim = feats["genre_sim"]
        m = artifacts.load_models()
        self.hybrid, self.M = m["hybrid"], m["interactions"]
        self.playlists = pd.read_csv(config.PLAYLISTS)
        self._search = (self.tracks["track_name"] + " " + self.tracks["artists"].str.replace(";", " ")).map(_fold).values
        self.n = len(self.tracks)

    # ---------- lookup ----------
    def search(self, query, k=10):
        q = _fold(query).split()
        if not q:
            return []
        hits = [i for i, s in enumerate(self._search) if all(w in s for w in q)]
        qs = " ".join(q)
        hits.sort(key=lambda i: (not self._search[i].startswith(qs), -self.tracks.popularity.iat[i]))
        return hits[:k]

    def describe(self, i):
        t = self.tracks.iloc[i]
        return {"title": t.track_name, "artist": t.artists.replace(";", ", "), "album": t.album_name,
                "genre": ", ".join(t.genres), "bpm": round(t.tempo), "key": t.key_name,
                "popularity": int(t.popularity), "url": f"https://open.spotify.com/track/{t.track_id}"}

    # ---------- recommending ----------
    def _seed_matrix(self, seeds):
        return sp.csr_matrix((np.ones(len(seeds), np.float32), (np.zeros(len(seeds), int), seeds)), shape=(1, self.n))

    def recommend(self, seeds, k=10, method="hybrid", diversity=0.2, max_per_artist=2, exclude_seed_artists=False):
        X = self._seed_matrix(seeds)
        h = self.hybrid
        if method == "hybrid":
            s = h.score(X)[0]
        else:
            s = {"content": h.content, "knn": h.knn, "als": h.als, "popularity": h.pop}[method].score(X)[0]
        s = np.asarray(s, dtype=np.float64).copy()
        s[seeds] = -np.inf
        seed_artists = set(a for i in seeds for a in self.tracks.artist_list.iat[i])
        return self._rerank(s, k, diversity, max_per_artist, seed_artists if exclude_seed_artists else set())

    def mood(self, targets: dict, genre=None, k=10, diversity=0.1):
        """targets: raw feature values, e.g. {"energy": 0.8, "valence": 0.7, "tempo": 125}."""
        t = self.tracks
        cols = [c for c in MOOD_DIMS if c in targets]
        X = t[cols].astype(float)
        mu, sd = X.mean(), X.std()
        d = (((X - mu) / sd - (pd.Series(targets)[cols] - mu) / sd) ** 2).sum(1).values
        s = -d + 0.15 * zscore(np.log1p(t.popularity.values)[None])[0]
        if genre:
            s = np.where(t.genres.map(lambda g: genre in g).values, s, -np.inf)
        return self._rerank(s, k, diversity, 2, set())

    def _rerank(self, s, k, diversity, max_per_artist, banned_artists):
        """Maximal marginal relevance: trade a little score for tracks unlike those already picked."""
        pool = np.argsort(-s)[:300]
        pool = [i for i in pool if np.isfinite(s[i])]
        base = (s[pool] - s[pool].mean()) / (s[pool].std() + 1e-9) if pool else []
        rel = dict(zip(pool, base))
        picked, per_artist = [], {}
        while len(picked) < k and pool:
            if picked and diversity > 0:
                sims = self.E[pool] @ self.E[picked].T
                mmr = np.array([rel[i] for i in pool]) - diversity * 4 * sims.max(1)
                j = int(np.argmax(mmr))
            else:
                j = 0
            i = pool.pop(j)
            arts = self.tracks.artist_list.iat[i]
            if banned_artists.intersection(arts) or per_artist.get(arts[0], 0) >= max_per_artist:
                continue
            per_artist[arts[0]] = per_artist.get(arts[0], 0) + 1
            picked.append(i)
        return [(i, float(s[i])) for i in picked]

    # ---------- explaining ----------
    def explain(self, seeds, i):
        """Short human reasons why track i was recommended for these seeds."""
        t = self.tracks
        best = max(seeds, key=lambda s_: float(self.E[s_] @ self.E[i]))
        reasons = []
        shared = sorted(set(t.genres.iat[i]) & set(t.genres.iat[best]))
        if shared:
            reasons.append(f"also {shared[0]}")
        else:
            g1, g2 = t.genres.iat[i][0], t.genres.iat[best][0]
            if self.genre_sim.loc[g1, g2] > 0.45:
                reasons.append(f"{g1} is close to {g2}")
        co = int(self.M[:, best].multiply(self.M[:, i]).sum())
        if co:
            reasons.append(f"in {co} playlist{'s' if co > 1 else ''} with {self._short(t.track_name.iat[best])}")
        a, b = t.iloc[i], t.iloc[best]
        traits = {"energy": "energy", "valence": "mood", "danceability": "groove", "acousticness": "acoustic feel"}
        diffs = sorted(traits, key=lambda c: abs(a[c] - b[c]))
        reasons.append(f"similar {traits[diffs[0]]}")
        if abs(a.tempo - b.tempo) < 6:
            reasons.append(f"{round(a.tempo)} BPM")
        return reasons

    @staticmethod
    def _short(name, n=28):
        name = re.sub(r"\s*[-(\[].*$", "", name) or name
        return name if len(name) <= n else name[: n - 1] + "…"

    # ---------- simulated listeners ----------
    def playlist_tracks(self, pid):
        return list(self.M[pid].indices)
