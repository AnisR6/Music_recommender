"""Top-K ranking metrics for implicit-feedback recommenders."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def evaluate(model, seeds: sp.csr_matrix, targets: sp.csr_matrix, ks=(10, 20), candidates=None,
             exclude: sp.csr_matrix = None, embeddings=None, popularity=None, batch=500, users=None,
             scores_for=None):
    """Score `seeds` rows with `model.score`, rank, and compare with `targets`.

    candidates: optional item index array to rank within (e.g. only new releases).
    exclude: items to never recommend per user (defaults to the seeds themselves).
    scores_for: optional callable(user_indices) -> scores, to reuse precomputed scores.
    Returns mean Recall@k, NDCG@k, HitRate@k, plus catalog coverage, novelty
    (mean popularity of recommended tracks) and intra-list diversity at max(k).
    """
    exclude = seeds if exclude is None else exclude
    users = np.where(np.diff(targets.indptr) > 0)[0] if users is None else users
    K = max(ks)
    res = {f"{m}@{k}": [] for k in ks for m in ("recall", "ndcg", "hit")}
    recommended, novelty, ild = set(), [], []
    discount = 1 / np.log2(np.arange(2, K + 2))

    for start in range(0, len(users), batch):
        u = users[start:start + batch]
        S = np.asarray(scores_for(u) if scores_for else model.score(seeds[u]), dtype=np.float32)
        S[exclude[u].nonzero()] = -np.inf
        if candidates is not None:
            mask = np.full(S.shape[1], -np.inf, np.float32); mask[candidates] = 0
            S = S + mask
        top = topk(S, K)
        T = targets[u]
        for row, items in enumerate(top):
            rel = set(T.indices[T.indptr[row]:T.indptr[row + 1]])
            hits = np.array([i in rel for i in items], float)
            for k in ks:
                h = hits[:k]
                res[f"recall@{k}"].append(h.sum() / min(k, len(rel)))
                ideal = discount[:min(k, len(rel))].sum()
                res[f"ndcg@{k}"].append((h * discount[:k]).sum() / ideal)
                res[f"hit@{k}"].append(float(h.any()))
            recommended.update(items[:10])
            if popularity is not None:
                novelty.append(popularity[items[:10]].mean())
            if embeddings is not None:
                E = embeddings[items[:10]]
                sim = E @ E.T
                ild.append(1 - (sim.sum() - np.trace(sim)) / (len(items[:10]) * (len(items[:10]) - 1)))

    out = {k: float(np.mean(v)) for k, v in res.items()}
    pool = len(candidates) if candidates is not None else seeds.shape[1]
    out["coverage@10"] = len(recommended) / pool
    if novelty: out["avg_popularity@10"] = float(np.mean(novelty))
    if ild: out["diversity@10"] = float(np.mean(ild))
    out["users"] = len(users)
    return out


def compare(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results).T.drop(columns="users", errors="ignore").round(4)
