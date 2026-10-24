"""Sparse playlist x track matrix and leakage-safe evaluation splits."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config


class InteractionMatrixBuilder:
    """Maps playlist ids and track URIs to row/column indices.

    Columns follow the catalog order (not only tracks that appear in a playlist),
    so content embeddings and CF scores line up column for column, and tracks
    nobody has added yet still have a column the content model can score.
    """

    def __init__(self, track_uris):
        self.track_to_idx = {u: i for i, u in enumerate(track_uris)}
        self.idx_to_track = dict(enumerate(track_uris))
        self.playlist_to_idx = {}

    def build(self, interactions: pd.DataFrame) -> sp.csr_matrix:
        pids = interactions["pid"].astype("category")
        self.playlist_to_idx = {p: i for i, p in enumerate(pids.cat.categories)}
        rows = pids.cat.codes.values
        cols = interactions["track_uri"].map(self.track_to_idx).values
        M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(self.playlist_to_idx), len(self.track_to_idx)))
        M.data[:] = 1.0  # a track counts once per playlist
        return M


def make_splits(M: sp.csr_matrix, cold_fraction=config.COLD_TRACK_FRACTION,
                val_fraction=config.VAL_FRACTION, test_fraction=config.TEST_FRACTION, seed=config.SEED):
    """Three evaluation settings in one pass.

    * cold items: a random slice of the catalog is treated as brand-new releases.
      Every interaction with them is removed from training and kept as `cold_test`.
    * warm items: each playlist's remaining tracks are split into train / val / test,
      so every playlist keeps some history (unlike a global random split, which can
      leave a playlist with nothing to learn from).
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = M.shape
    cold_items = np.sort(rng.choice(n_items, int(cold_fraction * n_items), replace=False))
    is_cold = np.zeros(n_items, bool); is_cold[cold_items] = True

    parts = {k: ([], []) for k in ("train", "val", "test", "cold_test")}
    for u in range(n_users):
        items = M.indices[M.indptr[u]:M.indptr[u + 1]]
        cold, warm = items[is_cold[items]], rng.permutation(items[~is_cold[items]])
        n_test = max(1, int(round(test_fraction * len(warm))))
        n_val = max(1, int(round(val_fraction * len(warm))))
        groups = {"test": warm[:n_test], "val": warm[n_test:n_test + n_val],
                  "train": warm[n_test + n_val:], "cold_test": cold}
        for k, it in groups.items():
            parts[k][0].extend([u] * len(it)); parts[k][1].extend(it)

    out = {k: sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=M.shape) for k, (r, c) in parts.items()}
    out["cold_items"] = cold_items
    return out


def save_splits(path, splits):
    np.savez_compressed(path, cold_items=splits["cold_items"],
                        **{f"{k}_{a}": getattr(splits[k], a) for k in ("train", "val", "test", "cold_test")
                           for a in ("data", "indices", "indptr")},
                        shape=np.array(splits["train"].shape))


def load_splits(path=config.SPLITS):
    z = np.load(path)
    shape = tuple(z["shape"])
    out = {k: sp.csr_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]), shape=shape)
           for k in ("train", "val", "test", "cold_test")}
    out["cold_items"] = z["cold_items"]
    return out


def short_playlists(train: sp.csr_matrix, n_seeds=3, min_items=10, seed=config.SEED):
    """Playlist cold start: keep only `n_seeds` random training tracks per playlist.

    Mimics a listener who has just started a playlist. Scored against the same
    held-out tracks as the full-history evaluation.
    """
    rng = np.random.default_rng(seed)
    rows, cols, users = [], [], []
    for u in range(train.shape[0]):
        items = train.indices[train.indptr[u]:train.indptr[u + 1]]
        if len(items) >= min_items:
            keep = rng.choice(items, n_seeds, replace=False)
            rows.extend([u] * n_seeds); cols.extend(keep); users.append(u)
    S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=train.shape)
    return S, np.array(users)
