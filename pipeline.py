import pandas as pd
import numpy as np
from src.features.content_embeddings import SpotifyTrackVectorizer
from src.features.interaction_matrix import InteractionMatrixBuilder
from src.models.collaborative import SpotifyHybridCollaborative
from src.models.fallback_handler import SpotifyColdStartRouter


def main():
    print("Initializing Spotify MPD Hybrid Recommender Pipeline...")

    # 1. Generate Synthetic Sample of MPD Structure for Verification
    print("Creating sample playlist-track data...")
    data = {
        'pid': [0, 0, 1, 1, 2, 2, 3],
        'track_uri': ['spotify:track:101', 'spotify:track:102', 'spotify:track:101',
                      'spotify:track:103', 'spotify:track:102', 'spotify:track:104', 'spotify:track:105'],
        'track_name': ['Song A', 'Song B', 'Song A', 'Song C', 'Song B', 'Song D', 'Song E'],
        'artist_name': ['Artist X', 'Artist Y', 'Artist X', 'Artist Z', 'Artist Y', 'Artist W', 'Artist V'],
        'album_name': ['Album 1', 'Album 2', 'Album 1', 'Album 3', 'Album 2', 'Album 4', 'Album 5'],
        'popularity': [85, 90, 85, 40, 90, 60, 10]
    }
    df = pd.DataFrame(data)

    # 2. Build Interaction Matrix
    matrix_builder = InteractionMatrixBuilder()
    coo_interactions = matrix_builder.build_coo_matrix(df)
    print(f"Interaction Matrix Shape: {coo_interactions.shape}")

    # 3. Generate SBERT Content Embeddings
    print("Generating dense SBERT vector embeddings for tracks...")
    vectorizer = SpotifyTrackVectorizer()
    unique_tracks = df[['track_uri', 'track_name', 'artist_name', 'album_name']].drop_duplicates()
    embeddings = vectorizer.generate_track_embeddings(unique_tracks)
    print(f"Generated Embeddings Matrix Shape: {embeddings.shape}")

    # 4. Train Hybrid Collaborative Model
    print("Fitting LightFM Collaborative Latent Space...")
    hybrid_model = SpotifyHybridCollaborative(no_components=16)
    hybrid_model.fit(coo_interactions, epochs=5)

    # 5. Execute Cold-Start Routing Engine
    router = SpotifyColdStartRouter(warm_playlist_threshold=2, warm_track_threshold=2)

    candidate_df = unique_tracks.copy()
    candidate_df['track_idx'] = [matrix_builder.track_to_idx[t] for t in candidate_df['track_uri']]
    candidate_df['interaction_count'] = [df[df['track_uri'] == t].shape[0] for t in candidate_df['track_uri']]
    candidate_df['popularity'] = [df[df['track_uri'] == t]['popularity'].values[0] for t in candidate_df['track_uri']]

    # Simulate recommendation routing for Playlist 0
    sim_matrix = np.dot(embeddings, embeddings.T)
    recs = router.route_recommendations(
        playlist_idx=0,
        playlist_track_count=2,
        candidate_tracks=candidate_df,
        hybrid_model=hybrid_model,
        content_similarity_matrix=sim_matrix
    )

    print("\nTop Recommended Track URIs for Playlist 0:")
    for idx, uri in enumerate(recs, 1):
        print(f"{idx}. {uri}")


if __name__ == "__main__":
    main()
