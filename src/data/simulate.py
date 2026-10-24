"""Simulated listeners and playlists built on the real track catalog.

Why simulate: the catalog has no user data. Spotify's Million Playlist Dataset
is the real thing but needs a gated sign-up. The original project built
playlists by chunking each genre by popularity and then sprinkling popular
tracks into 15 *random* playlists. That gives a collaborative model nothing
real to learn (its test AUC was 0.53, close to a coin flip).

This simulator gives every listener a taste the way real listening data has one:

* a home genre plus 0-2 neighbouring genres (picked from the learned genre map),
* a fixed "crossover" habit per genre (e.g. some metal listeners also play
  classical) that is consistent across listeners but invisible in the audio,
* a sound preference: a point in audio-feature space near their genres' average,
* a popularity appetite (hits only, or deep cuts),
* hidden taste factors shared with each artist (things like lyrics, scene or era
  that audio features cannot see),
* artist loyalty: once an artist is in a playlist, more of their songs follow.

So content features explain part of each playlist and co-listening patterns
explain the rest, which is the situation where a hybrid recommender should beat
either side alone. Every result built on these playlists is a result on
simulated behaviour, and the notebooks say so.
"""
import numpy as np
import pandas as pd

from src import config


def _pretty(genre):
    special = {"r-n-b": "R&B", "edm": "EDM", "idm": "IDM", "mpb": "MPB", "k-pop": "K-Pop", "j-pop": "J-Pop",
               "j-rock": "J-Rock", "j-dance": "J-Dance", "j-idol": "J-Idol", "hip-hop": "Hip-Hop", "dub": "Dub"}
    return special.get(genre, genre.replace("-", " ").title())


def simulate_playlists(tracks: pd.DataFrame, audio: np.ndarray, genre_sim: pd.DataFrame,
                       n_playlists=config.N_PLAYLISTS, k=config.LATENT_TASTE_DIM, seed=config.SEED):
    rng = np.random.default_rng(seed)
    n = len(tracks)
    genres = list(genre_sim.index)
    gidx = {g: i for i, g in enumerate(genres)}
    S = genre_sim.values.copy()
    np.fill_diagonal(S, 0)

    members = [[] for _ in genres]
    for i, gs in enumerate(tracks["genres"]):
        for g in gs:
            members[gidx[g]].append(i)
    members = [np.array(m) for m in members]
    sizes = np.array([len(m) for m in members], float)

    # hidden taste: artist-level factors plus a little per-track noise
    artist_codes = tracks["main_artist"].astype("category").cat.codes.values
    artist_latent = rng.normal(size=(artist_codes.max() + 1, k))
    track_latent = 0.8 * artist_latent[artist_codes] + 0.6 * rng.normal(size=(n, k))

    # each genre has two fixed crossover partners, unrelated to how it sounds
    crossover = {g: rng.choice([x for x in range(len(genres)) if x != g], 2, replace=False) for g in range(len(genres))}

    pop = (tracks["popularity"].values + 1) / 101
    d2_scale = np.median(((audio - audio.mean(0)) ** 2).sum(1))
    by_artist = pd.Series(np.arange(n)).groupby(artist_codes).apply(np.array).to_dict()

    rows, meta = [], []
    for pid in range(n_playlists):
        home = rng.choice(len(genres), p=sizes / sizes.sum())
        chosen = {home: 1.0}
        for g in rng.choice(len(genres), rng.integers(0, 3), replace=False, p=S[home] ** 4 / (S[home] ** 4).sum()):
            chosen.setdefault(g, 0.6)
        if rng.random() < 0.4:
            chosen.setdefault(rng.choice(crossover[home]), 0.5)

        pool = np.unique(np.concatenate([members[g] for g in chosen]))
        gw = np.zeros(n)
        for g, w in chosen.items():
            gw[members[g]] = np.maximum(gw[members[g]], w)

        home_centroid = audio[members[home]].mean(0)
        taste = home_centroid + rng.normal(scale=0.5, size=audio.shape[1])
        alpha = rng.uniform(0.3, 2.0)
        u = rng.normal(size=k)

        d2 = ((audio[pool] - taste) ** 2).sum(1)
        w = gw[pool] * np.exp(-d2 / (0.6 * d2_scale)) * pop[pool] ** alpha * np.exp(1.0 * track_latent[pool] @ u / np.sqrt(k))
        w = w / w.sum()

        length = int(np.clip(rng.lognormal(np.log(35), 0.5), 10, 150))
        length = min(length, len(pool))
        picked, in_list = [], set()
        while len(picked) < length:
            if picked and rng.random() < 0.25:                     # artist loyalty
                a = artist_codes[picked[rng.integers(len(picked))]]
                cand = [t for t in by_artist[a] if t not in in_list]
                if cand:
                    t = cand[rng.integers(len(cand))]
                    picked.append(t); in_list.add(t)
                    continue
            if rng.random() < 0.03:                                 # an off-taste discovery
                t = rng.choice(n, p=pop ** alpha / (pop ** alpha).sum())
            else:
                t = pool[rng.choice(len(pool), p=w)]
            if t not in in_list:
                picked.append(t); in_list.add(t)

        g_names = [genres[g] for g in chosen]
        name = " & ".join(_pretty(x) for x in g_names[:2]) + f" Mix #{pid}"
        meta.append({"pid": pid, "playlist_name": name, "home_genre": genres[home],
                     "taste_genres": ";".join(g_names), "pop_appetite": round(alpha, 2), "n_tracks": len(picked)})
        rows.extend((pid, name, pos, t) for pos, t in enumerate(picked))

    inter = pd.DataFrame(rows, columns=["pid", "playlist_name", "pos", "track_idx"])
    t = tracks.iloc[inter["track_idx"]].reset_index(drop=True)
    inter = pd.concat([inter.drop(columns="track_idx"),
                       t[["track_uri", "track_name", "artists", "album_name", "popularity", "primary_genre"]]], axis=1)
    return inter, pd.DataFrame(meta)
