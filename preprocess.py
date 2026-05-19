import numpy as np
import os

def custom_preprocess(X_raw):
    N_samples = X_raw.shape[0]
    y = X_raw[:, 5].reshape(-1, 1).astype(np.float64)
    X_processed = np.zeros((N_samples, 19), dtype=np.float64)
    
    age = X_raw[:, 2].astype(np.float64)
    X_processed[:, 0] = (age - np.min(age)) / (np.max(age) - np.min(age))
    
    X_processed[:, 1] = X_raw[:, 3].astype(np.float64)
    
    tt = X_raw[:, 4].astype(np.float64)
    X_processed[:, 2] = (tt - np.min(tt)) / (np.max(tt) - np.min(tt))
    
    for i, col_idx in enumerate(range(7, 12)):
        jitter_col = X_raw[:, col_idx].astype(np.float64)
        log_jitter = np.log(jitter_col + 1e-7)
        X_processed[:, 3 + i] = (log_jitter - np.min(log_jitter)) / (np.max(log_jitter) - np.min(log_jitter))
        
    for i, col_idx in enumerate(range(12, 18)):
        shimmer_col = X_raw[:, col_idx].astype(np.float64)
        median = np.median(shimmer_col)
        q75, q25 = np.percentile(shimmer_col, [75, 25])
        iqr = q75 - q25
        iqr = iqr if iqr > 0 else 1e-7
        X_processed[:, 8 + i] = (shimmer_col - median) / iqr
        
    nhr = X_raw[:, 18].astype(np.float64)
    X_processed[:, 14] = np.log(nhr + 1e-7)
    
    hnr = X_raw[:, 19].astype(np.float64)
    X_processed[:, 15] = (hnr - np.mean(hnr)) / np.std(hnr)
    
    for i, col_idx in enumerate(range(20, 23)):
        nonlinear_col = X_raw[:, col_idx].astype(np.float64)
        X_processed[:, 16 + i] = (nonlinear_col - np.min(nonlinear_col)) / (np.max(nonlinear_col) - np.min(nonlinear_col))
        
    return X_processed, y

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    
    print("Fetching raw OpenML dataset...")
    X_raw, _ = fetch_openml(data_id=44964, return_X_y=True, as_frame=False, parser='auto')
    
    print("Running custom data engineering transformations...")
    X, y = custom_preprocess(X_raw)
    
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    
    x_path = os.path.join(cache_dir, "parkinsons_X_processed.npy")
    y_path = os.path.join(cache_dir, "parkinsons_y_processed.npy")
    
    np.save(x_path, X)
    np.save(y_path, y)
    
    print(f"Success! Processed data written to disk:")
    print(f" -> {x_path} {X.shape}")
    print(f" -> {y_path} {y.shape}")