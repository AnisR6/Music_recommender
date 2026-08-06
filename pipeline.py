import os
import pandas as pd
import numpy as np
from src.features.content_embeddings import SpotifyTrackVectorizer
from src.features.interaction_matrix import InteractionMatrixBuilder
from src.models.collaborative import SpotifyHybridCollaborative
from src.models.fallback_handler import SpotifyColdStartRouter

PROCESSED_INTERACTIONS = os.path.join('data', 'processed', 'playlist_track_interactions.csv')


def load_playlist_track_data() -> pd.DataFrame:
    """
    Loads the real playlist-track interaction table built by
    notebooks/02_preprocessing.ipynb if it's been generated, and falls
    back to a tiny synthetic sample otherwise so this script still runs
    standalone (e.g. as a smoke test) without requiring the notebooks
    to be run first.
    """
    if os.path.exists(PROCESSED_INTERACTIONS):
        print(f"Loading real processed data from {PROCESSED_INTERACTIONS}...")
        df = pd.read_csv(PROCESSED_INTERACTIONS)
        return df

    print("No processed data found - run the notebooks/ pipeline first for real "
          "data. Falling back to a small synthetic sample for now.")
    data = {
        'pid': [0, 0, 1, 1, 2, 2, 3],
        'track_uri': ['spotify:track:101', 'spotify:track:102', 'spotify:track:101',
                      'spotify:track:103', 'spotify:track:102', 'spotify:track:104', 'spotify:track:105'],
        'track_name': ['Song A', 'Song B', 'Song A', 'Song C', 'Song B', 'Song D', 'Song E'],
        'artist_name': ['Artist X', 'Artist Y', 'Artist X', 'Artist Z', 'Artist Y', 'Artist W', 'Artist V'],
        'album_name': ['Album 1', 'Album 2', 'Album 1', 'Album 3', 'Album 2', 'Album 4', 'Album 5'],
        'popularity': [85, 90, 85, 40, 90, 60, 10]
    }
    return pd.DataFrame(data)


def main():
    print("Initializing Spotify MPD Hybrid Recommender Pipeline...")

    df = load_playlist_track_data()
    print(f"Loaded {len(df)} playlist-track rows, "
          f"{df['pid'].nunique()} playlists, {df['track_uri'].nunique()} unique tracks.")

    # 1. Build Interaction Matrix
    matrix_builder = InteractionMatrixBuilder()
    coo_interactions = matrix_builder.build_coo_matrix(df)
    print(f"Interaction Matrix Shape: {coo_interactions.shape}")

    # 2. Generate SBERT Content Embeddings
    print("Generating dense SBERT vector embeddings for tracks...")
    vectorizer = SpotifyTrackVectorizer()
    unique_tracks = df[['track_uri', 'track_name', 'artist_name', 'album_name']].drop_duplicates()
    embeddings = vectorizer.generate_track_embeddings(unique_tracks)
    print(f"Generated Embeddings Matrix Shape: {embeddings.shape}")

    # 3. Train Hybrid Collaborative Model
    print("Fitting LightFM Collaborative Latent Space...")
    hybrid_model = SpotifyHybridCollaborative(no_components=32)
    hybrid_model.fit(coo_interactions, epochs=15)

    # 4. Execute Cold-Start Routing Engine
    router = SpotifyColdStartRouter(warm_playlist_threshold=5, warm_track_threshold=10)

    candidate_df = unique_tracks.copy()
    candidate_df['track_idx'] = [matrix_builder.track_to_idx[t] for t in candidate_df['track_uri']]
    interaction_counts = df.groupby('track_uri').size()
    candidate_df['interaction_count'] = candidate_df['track_uri'].map(interaction_counts).fillna(0).astype(int)
    if 'popularity' in df.columns:
        first_popularity = df.drop_duplicates('track_uri').set_index('track_uri')['popularity']
        candidate_df['popularity'] = candidate_df['track_uri'].map(first_popularity)
    else:
        candidate_df['popularity'] = candidate_df['interaction_count']

    # 5. Route recommendations for the first playlist in the data
    target_pid = df['pid'].iloc[0]
    playlist_tracks = df[df['pid'] == target_pid]['track_uri'].tolist()
    playlist_track_indices = [matrix_builder.track_to_idx[t] for t in playlist_tracks]

    sim_matrix = np.dot(embeddings, embeddings.T)
    recs = router.route_recommendations(
        playlist_idx=matrix_builder.playlist_to_idx[target_pid],
        playlist_track_count=len(playlist_tracks),
        candidate_tracks=candidate_df,
        hybrid_model=hybrid_model,
        content_similarity_matrix=sim_matrix,
        playlist_track_indices=playlist_track_indices
    )

    print(f"\nTop Recommended Track URIs for Playlist {target_pid}:")
    for idx, uri in enumerate(recs, 1):
        print(f"{idx}. {uri}")


if __name__ == "__main__":
    main()
