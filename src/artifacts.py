"""Save and load everything the console app needs."""
import pickle

import numpy as np
import pandas as pd

from src import config


def save_features(audio, audio_cols, genre_vecs, text_vecs, embeddings, genre_sim: pd.DataFrame,
                  block_weights, path=config.FEATURES):
    np.savez_compressed(path, audio=audio, audio_cols=np.array(list(audio_cols), dtype=str), genre=genre_vecs, text=text_vecs,
                        embeddings=embeddings, genre_sim=genre_sim.values, genre_names=np.array(list(genre_sim.index), dtype=str),
                        block_weights=np.array([block_weights[k] for k in ("audio", "genre", "text")]))


def load_features(path=config.FEATURES):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["genre_sim"] = pd.DataFrame(out["genre_sim"], index=out["genre_names"], columns=out["genre_names"])
    out["block_weights"] = dict(zip(("audio", "genre", "text"), out["block_weights"]))
    return out


def save_models(models: dict, path=config.MODELS):
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load_models(path=config.MODELS):
    with open(path, "rb") as f:
        return pickle.load(f)
