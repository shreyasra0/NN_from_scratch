import os
import numpy as np
from sklearn.datasets import fetch_openml

def load_parkinsons_data(normalize=False):
    cache_dir = "data"
    suffix = "_norm" if normalize else "_raw"
    x_cache = os.path.join(cache_dir, f"parkinsons_X{suffix}.npy")
    y_cache = os.path.join(cache_dir, f"parkinsons_y{suffix}.npy")
    
    if os.path.exists(x_cache) and os.path.exists(y_cache):
        return np.load(x_cache), np.load(y_cache)

    os.makedirs(cache_dir, exist_ok=True)
    
    X_raw, y_raw = fetch_openml(data_id=44964, return_X_y=True, as_frame=False, parser='auto')
    
    X = X_raw[:, 6:22].astype(np.float64)
    y = X_raw[:, 4].reshape(-1, 1).astype(np.float64)
    
    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
        
    np.save(x_cache, X)
    np.save(y_cache, y)
    
    return X, y

if __name__ == "__main__":
    X, y = load_parkinsons_data(normalize=False)