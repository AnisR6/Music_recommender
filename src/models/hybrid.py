"""Hybrid recommender with soft cold-start routing.

The original router used hard thresholds: a playlist with < 5 tracks got pure
popularity, a track with < 10 plays got pure content, everything else pure
LightFM. Hard switches throw information away right at the boundary (a track
with 9 plays ignores all 9). Here every track gets a CF confidence

    c_i = n_i / (n_i + kappa)        n_i = playlists containing track i in training

and the final score blends the parts by that confidence:

    score = c_i * c_u * CF + w_content * Content + w_pop * Popularity

so a brand-new release (n_i = 0) is ranked by content alone, a hit by mostly CF,
and everything in between by a mix. The same idea applies to the playlist side:
c_u = e_u / (e_u + kappa_user), where e_u is how much co-listening evidence the
seed tracks carry (the sum of their n_i). A playlist seeded with one rarely
added song leans on content; one seeded with well-known songs leans on CF. Each component is z-scored per playlist so
the weights are comparable.
"""
import numpy as np


def zscore(S):
    S = np.asarray(S, dtype=np.float32)
    mu = S.mean(1, keepdims=True)
    sd = S.std(1, keepdims=True) + 1e-6
    return (S - mu) / sd


class HybridRecommender:
    name = "Hybrid"

    def __init__(self, knn, als, content, popularity, kappa=5.0, cf_mix=0.5, w_content=0.4, w_pop=0.1,
                 kappa_user=20.0):
        self.knn, self.als, self.content, self.pop = knn, als, content, popularity
        self.kappa, self.cf_mix, self.w_content, self.w_pop = kappa, cf_mix, w_content, w_pop
        self.kappa_user = kappa_user

    def fit(self, train):
        n = np.asarray(train.sum(0)).ravel()
        self.n_i = n
        return self

    def set_params(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        return self

    def components(self, seeds):
        return {"knn": zscore(self.knn.score(seeds)), "als": zscore(self.als.score(seeds)),
                "content": zscore(self.content.score(seeds)), "pop": zscore(self.pop.score(seeds)),
                "evidence": np.asarray(seeds @ self.n_i, dtype=np.float32).ravel()}

    def blend(self, comp):
        c_i = (self.n_i / (self.n_i + self.kappa)).astype(np.float32)[None, :]
        e = comp["evidence"][:, None]
        c_u = e / (e + self.kappa_user)
        cf = self.cf_mix * comp["knn"] + (1 - self.cf_mix) * comp["als"]
        return c_i * c_u * cf + self.w_content * comp["content"] + self.w_pop * comp["pop"]

    def score(self, seeds):
        # a playlist with no seeds at all is pure popularity, the only signal left
        S = self.blend(self.components(seeds))
        empty = np.diff(seeds.indptr) == 0
        if empty.any():
            S[empty] = self.pop.score(seeds[empty])
        return S


def tune(hybrid, seeds, targets, users, grid, metric="recall@20", candidates=None):
    """Grid-search the blend weights on a validation split.

    Component scores are computed once for the validation playlists and
    re-blended for every combination, so the search costs one scoring pass.
    """
    import itertools
    import pandas as pd
    from src.evaluation.metrics import evaluate

    comp = hybrid.components(seeds[users])
    pos = {u: i for i, u in enumerate(users)}
    rows = []
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        hybrid.set_params(**dict(zip(keys, values)))
        def scores_for(u):
            idx = [pos[x] for x in u]
            return hybrid.blend({k: v[idx] for k, v in comp.items()})
        r = evaluate(hybrid, seeds, targets, users=users, scores_for=scores_for, candidates=candidates, ks=(20,))
        rows.append({**dict(zip(keys, values)), metric: r[metric]})
    table = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    hybrid.set_params(**{k: table.loc[0, k] for k in keys})
    return table


class HardThresholdRouter:
    """The original project's router, kept as a baseline for comparison.

    playlist with < `warm_playlist` tracks -> popularity
    track with < `warm_track` plays        -> content similarity
    otherwise                               -> collaborative score
    The original also mixed raw LightFM scores and cosine similarities in one
    ranking even though they live on different scales; this keeps that behaviour.
    """
    name = "Hard-threshold router (original)"

    def __init__(self, cf, content, popularity, warm_playlist=5, warm_track=10):
        self.cf, self.content, self.pop = cf, content, popularity
        self.warm_playlist, self.warm_track = warm_playlist, warm_track

    def fit(self, train):
        self.n_i = np.asarray(train.sum(0)).ravel()
        return self

    def score(self, seeds):
        cf, ct = self.cf.score(seeds), self.content.score(seeds)
        S = np.where(self.n_i[None, :] >= self.warm_track, cf, ct).astype(np.float32)
        cold_playlist = np.diff(seeds.indptr) < self.warm_playlist
        if cold_playlist.any():
            S[cold_playlist] = self.pop.score(seeds[cold_playlist])
        return S
