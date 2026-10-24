from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
FIGURES = PROJECT_ROOT / "notebooks" / "figures"

RAW_TRACKS = DATA_RAW / "spotify_tracks_dataset.csv"
TRACKS_CLEAN = DATA_PROCESSED / "tracks_clean.parquet"
INTERACTIONS = DATA_PROCESSED / "playlist_track_interactions.csv"
PLAYLISTS = DATA_PROCESSED / "playlists.csv"
FEATURES = DATA_PROCESSED / "content_features.npz"
SPLITS = DATA_PROCESSED / "splits.npz"
MODELS = DATA_PROCESSED / "models.pkl"

SEED = 42

# Catalog
TRACKS_PER_GENRE_CAP = 200          # most popular N per genre (None = full ~81k catalog)

# Listener simulation (notebook 02)
N_PLAYLISTS = 6000
LATENT_TASTE_DIM = 8                # hidden taste factors that audio features cannot see

# Splits (notebook 04)
COLD_TRACK_FRACTION = 0.08          # "new releases" never seen in training
TEST_FRACTION = 0.2
VAL_FRACTION = 0.1

# Models (defaults = the values tuned on validation in notebook 05)
ALS_FACTORS = 128
ALS_REG = 0.2
ALS_ITERS = 20
ALS_ALPHA = 60.0
KNN_NEIGHBOURS = 200

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo"]
