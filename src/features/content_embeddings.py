import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

class SpotifyTrackVectorizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def generate_track_embeddings(self, tracks_df: pd.DataFrame) -> np.ndarray:
        """
        Combines Spotify track metadata (track_name, artist_name, album_name)
        into a single text string and generates L2-normalized dense embeddings.
        """
        combined_text = (
            tracks_df['track_name'].fillna('') + ' by ' + 
            tracks_df['artist_name'].fillna('') + ' from album ' + 
            tracks_df['album_name'].fillna('')
        )
        embeddings = self.model.encode(
            combined_text.tolist(), 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        return normalize(embeddings, norm='l2', axis=1)