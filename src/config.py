from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "spotify_global_2019_most_streamed_tracks_audio_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
    "tempo",
    "time_signature",
    "duration_ms",
    "key",
    "mode",
    "Artist_popularity",
    "Artist_follower",
]