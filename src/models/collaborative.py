import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix

class SpotifyHybridCollaborative:
    def __init__(self, no_components: int = 64, loss: str = 'warp', lr: float = 0.05):
        self.model = LightFM(no_components=no_components, loss=loss, learning_rate=lr)
        
    def fit(self, interactions: coo_matrix, item_features: coo_matrix = None, epochs: int = 15):
        \"\"\"
        Trains the hybrid LightFM model using WARP loss for playlist track ranking.
        \"\"\"
        self.model.fit(
            interactions=interactions,
            item_features=item_features,
            epochs=epochs,
            num_threads=4,
            verbose=True
        )
        
    def predict_cold_track(self, playlist_indices: np.ndarray, track_idx: int, 
                           item_features: coo_matrix) -> np.ndarray:
        \"\"\"
        Predicts addition probability for a newly released track across target playlists
        using solely its auxiliary content features.
        \"\"\"
        return self.model.predict(
            user_ids=playlist_indices,
            item_ids=np.repeat(track_idx, len(playlist_indices)),
            item_features=item_features)