# Music Recommender

A hybrid music recommender built on the open **Spotify Tracks Dataset** (114,000 tracks,
114 genres, BSD licence), with the full data-science workflow in seven notebooks and a small
console app on top.

```
python cli.py similar "clair de lune"
python cli.py mix "get lucky" "september" "billie jean"
python cli.py mood --energy 0.2 --valence 0.3 --acousticness 0.9
python cli.py            # interactive menu
```

## Quick start

```bash
pip install -r requirements.txt
python pipeline.py       # builds features, trains and evaluates all models (~3 min)
python cli.py
```

Or run the notebooks in order (`notebooks/01` → `07`); notebook 07 saves the same models the
console app loads. Python 3.10-3.13.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda.ipynb` | Data quality, multi-genre labels, duplicate releases, tempo octave errors, skewed features, correlations, genre audio profiles and a PCA map of genres |
| 02 | `02_preprocessing.ipynb` | One row per song: genres merged into lists, same-song duplicates merged, failed analyses dropped, catalog capped to 200 songs per genre (22,800 songs) |
| 03 | `03_feature_engineering.ipynb` | Tempo folding, circle-of-fifths key encoding, quantile transforms, derived features and weights; a learned genre embedding (PPMI co-listing + sound); TF-IDF text block; combined content embedding |
| 04 | `04_interaction_matrix.ipynb` | Simulated listeners with a documented taste model, the sparse playlist × track matrix, and leakage-safe splits (warm, new releases) |
| 05 | `05_models.ipynb` | Content-model ablation and block-weight tuning; item-kNN, ALS and BPR with hyperparameter grids, all on validation |
| 06 | `06_hybrid_cold_start.ipynb` | Hybrid with soft, confidence-weighted cold-start routing, tuned on validation and compared with the original hard-threshold router |
| 07 | `07_evaluation_summary.ipynb` | One-time test evaluation in three settings, accuracy vs coverage / popularity / diversity, results by playlist length, example recommendations, final model export |

Figures are saved to `notebooks/figures/`.

## How it recommends

1. **Content embedding** per track: engineered audio features + a learned 24-d genre vector + a
   small TF-IDF text block (titles, albums, artists), each normalised and weighted.
2. **Collaborative filtering** on playlists: item-kNN (cosine with shrinkage) and ALS
   (`implicit`). Both fold in a new playlist from a few seed songs.
3. **Hybrid**: each component is z-scored, then
   `score = c_i · c_u · CF + w_content · Content + w_pop · Popularity`,
   where `c_i = n_i / (n_i + κ)` trusts CF only as much as a track has been played and
   `c_u` does the same for the seed songs. New releases fall back to content automatically.
4. **Re-ranking** in the app: maximal marginal relevance for variety, at most two songs per artist.

## About the data

The tracks, audio features, genres and popularity are real. The dataset has **no users or
playlists**, so notebook 04 simulates 6,000 listeners (home genre, neighbouring genres, fixed
crossover habits, a sound preference, a popularity appetite, hidden artist-level tastes and
artist loyalty). Collaborative and hybrid results are results on that simulated behaviour. To use
real listening data, write `pid, track_uri` rows (for example from Spotify's Million Playlist
Dataset) to `data/processed/playlist_track_interactions.csv` and re-run notebooks 04-07.

## Layout

```
cli.py                  console app
pipeline.py             everything the notebooks do, headless
src/config.py           paths and settings
src/data/               loading, cleaning, listener simulation
src/features/           audio, genre, text features; content embedding; interaction matrix and splits
src/models/             baselines, content, item-kNN / ALS / BPR, hybrid and the original router
src/evaluation/         ranking metrics
src/recommender.py      the object the console app talks to
data/raw/               spotify_tracks_dataset.csv
data/processed/         generated artifacts
```
