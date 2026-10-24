"""Load and clean the Spotify Tracks catalog."""
import numpy as np
import pandas as pd

from src import config

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def song_key(names: pd.Series, artists: pd.Series) -> pd.Series:
    """Normalised "title|main artist" used to spot the same song under several ids.

    Drops version suffixes such as " - Remastered 2011" or " (feat. X)" but keeps
    remixes apart, since a remix is a different recording people add on purpose.
    """
    lower = names.str.lower()
    stripped = lower.str.replace(r"\s*[-(\[].*$", "", regex=True).str.strip()
    title = lower.where(lower.str.contains("remix"), stripped)
    return title + "|" + artists.str.lower().str.split(";").str[0]


def load_raw(path=config.RAW_TRACKS) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def clean_catalog(raw: pd.DataFrame, per_genre_cap=config.TRACKS_PER_GENRE_CAP) -> pd.DataFrame:
    """One row per song, with every genre it is listed under.

    The raw file has 114k rows but only ~90k distinct track ids, because a track
    listed under several genres appears once per genre. The original pipeline kept
    only the first genre. Here we keep them all as a multi-label list, then also
    collapse the same song released on several albums (same title + artist).
    """
    df = raw.dropna(subset=["track_name", "artists", "album_name"]).copy()

    genres = df.groupby("track_id")["track_genre"].agg(lambda s: sorted(set(s)))
    df = (df.sort_values("popularity", ascending=False)
            .drop_duplicates("track_id")
            .assign(genres=lambda d: d["track_id"].map(genres)))

    # same song on a single, an album and a compilation -> keep the most popular copy,
    # but merge the genre lists so no label is lost
    df["song_key"] = song_key(df["track_name"], df["artists"])
    merged = df.groupby("song_key")["genres"].agg(lambda s: sorted(set(g for gs in s for g in gs)))
    df = df.drop_duplicates("song_key").assign(genres=lambda d: d["song_key"].map(merged))

    # anomalies: tempo 0 / time signature 0 are failed analyses, not real values
    df = df[(df["tempo"] > 0) & (df["time_signature"] > 0) & (df["duration_ms"] > 30_000)]

    df["primary_genre"] = df["track_genre"]
    if per_genre_cap:
        df = (df.sort_values("popularity", ascending=False)
                .groupby("primary_genre", group_keys=False).head(per_genre_cap))

    df = df.sort_values("popularity", ascending=False).reset_index(drop=True)
    df["artist_list"] = df["artists"].str.split(";")
    df["main_artist"] = df["artist_list"].str[0]
    df["track_uri"] = "spotify:track:" + df["track_id"]
    df["key_name"] = df["key"].map(lambda k: KEY_NAMES[k] if k >= 0 else "?") + np.where(df["mode"] == 1, " major", " minor")
    keep = ["track_id", "track_uri", "track_name", "artists", "artist_list", "main_artist", "album_name",
            "popularity", "duration_ms", "explicit", "key", "mode", "key_name", "time_signature",
            *config.AUDIO_FEATURES, "primary_genre", "genres"]
    return df[keep]


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(config.TRACKS_CLEAN)
