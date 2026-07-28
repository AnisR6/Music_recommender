import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

class InteractionMatrixBuilder:
    def __init__(self):
        self.playlist_to_idx = {}
        self.track_to_idx = {}
        self.idx_to_track = {}
        
    def build_coo_matrix(self, interactions_df: pd.DataFrame) -> coo_matrix:
        """
        Maps Spotify playlist pids and track URIs to categorical indices
        and constructs a sparse COO interaction matrix.
        """
        playlists = interactions_df['pid'].astype('category')
        tracks = interactions_df['track_uri'].astype('category')
        
        self.playlist_to_idx = {val: idx for idx, val in enumerate(playlists.cat.categories)}
        self.track_to_idx = {val: idx for idx, val in enumerate(tracks.cat.categories)}
        self.idx_to_track = {idx: val for val, idx in self.track_to_idx.items()}
        
        row = playlists.cat.codes.values
        col = tracks.cat.codes.values
        data = np.ones(len(interactions_df), dtype=np.float32)
        
        return coo_matrix(
            (data, (row, col)), 
            shape=(len(self.playlist_to_idx), len(self.track_to_idx))
        )