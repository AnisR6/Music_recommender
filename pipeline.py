"""Run the whole pipeline without notebooks: data -> features -> models -> evaluation -> saved artifacts.

    python pipeline.py            # about 2-3 minutes on a laptop
    python cli.py                 # then use the console app

The notebooks walk through the same steps with plots and commentary.
"""
import itertools
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from src import artifacts, config
from src.data.load import clean_catalog, load_raw
from src.data.simulate import simulate_playlists
from src.evaluation.metrics import compare, evaluate
from src.features.audio_features import AudioFeatureEngineer, weighted
from src.features.content_embeddings import combine
from src.features.genre_embeddings import GenreEmbedder
from src.features.interaction_matrix import InteractionMatrixBuilder, make_splits, save_splits, short_playlists
from src.features.text_features import TextEmbedder
from src.models.baselines import PopularityRecommender
from src.models.collaborative import ALSRecommender, BPRRecommender, ItemKNN
from src.models.content import ContentRecommender
from src.models.hybrid import HybridRecommender, tune


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    step("1/6 cleaning catalog")
    tracks = clean_catalog(load_raw())
    tracks.to_parquet(config.TRACKS_CLEAN)
    print(f"      {len(tracks):,} tracks, {tracks.genres.explode().nunique()} genres")

    step("2/6 engineering features")
    fe = AudioFeatureEngineer()
    audio_df = fe.fit_transform(tracks)
    audio = weighted(audio_df)
    genre = GenreEmbedder().fit(tracks.genres, audio)
    text = TextEmbedder().fit_transform(tracks)
    blocks = {"audio": audio, "genre": genre.track_vectors(), "text": text}

    step("3/6 simulating listeners")
    inter, meta = simulate_playlists(tracks, audio, genre.similarity_)
    inter.to_csv(config.INTERACTIONS, index=False)
    meta.to_csv(config.PLAYLISTS, index=False)
    builder = InteractionMatrixBuilder(tracks.track_uri.tolist())
    M = builder.build(inter)
    splits = make_splits(M)
    save_splits(config.SPLITS, splits)
    train, val, test, cold = splits["train"], splits["val"], splits["test"], splits["cold_test"]
    print(f"      {M.shape[0]:,} playlists, {M.nnz:,} playlist-track pairs")

    step("4/6 tuning content block weights on validation")
    rng = np.random.default_rng(0)
    val_users = rng.choice(np.where(np.diff(val.indptr) > 0)[0], 1500, replace=False)
    best = None
    for wa, wg, wt in itertools.product([1.0], [2, 4, 8, 16], [0.0, 0.1, 0.3]):
        w = {"audio": wa, "genre": wg, "text": wt}
        r = evaluate(ContentRecommender(combine(blocks, w)), train, val, users=val_users, ks=(20,))["recall@20"]
        if best is None or r > best[0]:
            best = (r, w)
    block_weights = best[1]
    E = combine(blocks, block_weights)
    print(f"      best weights {block_weights}  val recall@20 {best[0]:.4f}")
    artifacts.save_features(audio, audio_df.columns, blocks["genre"], text, E, genre.similarity_, block_weights)

    step("5/6 training and evaluating models")
    pop = PopularityRecommender().fit(train)
    content = ContentRecommender(E)
    knn = ItemKNN().fit(train)
    als = ALSRecommender().fit(train)
    bpr = BPRRecommender().fit(train)
    hybrid = HybridRecommender(knn, als, content, pop).fit(train)
    grid = {"kappa": [1, 2, 5, 15], "cf_mix": [0.3, 0.5, 0.7], "w_content": [0.25, 0.5, 1.0, 2.0], "w_pop": [0.0, 0.1]}
    table = tune(hybrid, train, val, val_users, grid)
    short, short_users = short_playlists(train)
    short_val = short_users[np.isin(short_users, val_users)]
    ku = tune(hybrid, short, val, short_val, {"kappa_user": [0, 5, 20, 60, 150, 400]})
    print(f"      hybrid params {table.iloc[0].to_dict()}, kappa_user {ku.iloc[0].kappa_user}")

    seen = train + val
    raw_audio = ContentRecommender(StandardScaler().fit_transform(tracks[config.AUDIO_FEATURES]), name="Content (raw audio)")
    text_only = ContentRecommender(text, name="Content (text only)")
    models = [pop, raw_audio, text_only, content, knn, als, bpr, hybrid]
    warm = {m.name: evaluate(m, train, test, exclude=seen, embeddings=E, popularity=tracks.popularity.values) for m in models}
    coldr = {m.name: evaluate(m, train, cold, exclude=seen, candidates=splits["cold_items"]) for m in models}
    shortr = {m.name: evaluate(m, short, test, exclude=seen, users=short_users) for m in models}
    print("\nWarm tracks (held-out 20% of each playlist):")
    print(compare(warm)[["recall@10", "recall@20", "ndcg@10", "hit@10", "coverage@10", "diversity@10"]].to_string())
    print("\nNew releases (tracks never seen in training):")
    print(compare(coldr)[["recall@10", "recall@20", "ndcg@10", "hit@10"]].to_string())

    print("\nShort playlists (only 3 seed tracks):")
    print(compare(shortr)[["recall@10", "recall@20", "ndcg@10", "hit@10"]].to_string())

    step("6/6 refitting on all playlists for the console app")
    pop_f = PopularityRecommender().fit(M)
    knn_f, als_f = ItemKNN().fit(M), ALSRecommender().fit(M)
    final = HybridRecommender(knn_f, als_f, content, pop_f, **{k: float(table.iloc[0][k]) for k in grid},
                               kappa_user=float(ku.iloc[0].kappa_user)).fit(M)
    artifacts.save_models({"hybrid": final, "interactions": M, "block_weights": block_weights,
                           "warm_results": warm, "cold_results": coldr, "short_results": shortr})
    step(f"done. Try: python cli.py")


if __name__ == "__main__":
    main()
