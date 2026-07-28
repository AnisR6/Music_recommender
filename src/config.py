import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

# Spotify MPD specific schema parameters
MPD_SLICE_PREFIX = "mpd.slice."
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NO_COMPONENTS = 64
WARP_EPOCHS = 15
WARM_PLAYLIST_THRESHOLD = 5
WARM_TRACK_THRESHOLD = 10