import numpy as np
import pandas as pd

class SpotifyColdStartRouter:
    def __init__(self, warm_playlist_threshold: int = 5, warm_track_threshold: int = 10):
        self.playlist_threshold = warm_playlist_threshold
        self.track_threshold = warm_track_threshold
        
    def route_recommendations(self, playlist_idx: int, playlist_track_count: int,
                              candidate_tracks: pd.DataFrame, hybrid_model,
                              content_similarity_matrix: np.ndarray,
                              playlist_track_indices: list = None) -> list:
        """
        Dynamically routes recommendation scoring based on playlist and track sparsity.

        `playlist_track_indices` should be the track_idx values of the tracks
        already in this playlist - used to score content-based similarity
        against what the playlist actually contains, rather than the whole
        catalog.
        """
        # 1. Playlist Cold-Start: Fallback to global track popularity
        if playlist_track_count < self.playlist_threshold:
            return self._get_popularity_fallback(candidate_tracks)

        playlist_track_indices = playlist_track_indices or []

        scores = []
        for _, track in candidate_tracks.iterrows():
            track_idx = track['track_idx']
            track_interaction_count = track['interaction_count']
            
            # 2. Track Cold-Start: Rely on SBERT content-similarity
            if track_interaction_count < self.track_threshold:
                score = self._compute_content_score(track_idx, playlist_track_indices, content_similarity_matrix)
            # 3. Warm Playlist + Warm Track: Use Collaborative Latent Space
            else:
                score = hybrid_model.model.predict(np.array([playlist_idx]), np.array([track_idx]))[0]
                
            scores.append((track['track_uri'], score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return [track_uri for track_uri, _ in scores[:20]]

    def _get_popularity_fallback(self, candidate_tracks: pd.DataFrame) -> list:
        sorted_tracks = candidate_tracks.sort_values(by='popularity', ascending=False)
        return sorted_tracks['track_uri'].head(20).tolist()

    def _compute_content_score(self, track_idx: int, playlist_track_indices: list, sim_matrix: np.ndarray) -> float:
        """
        Mean content similarity between a candidate track and the tracks
        already in the target playlist (not the whole catalog).
        """
        if not playlist_track_indices:
            return float(np.mean(sim_matrix[track_idx]))
        return float(np.mean(sim_matrix[track_idx, playlist_track_indices]))