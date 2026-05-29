import numpy as np, os, joblib
from sklearn.preprocessing import StandardScaler

def make_windows(series: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)
 
def scale_series(arr: np.ndarray, out_path: str):
    scaler = StandardScaler()
    arr2d = arr.reshape(-1, 1)
    scaled = scaler.fit_transform(arr2d).flatten()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(scaler, out_path)
    return scaled, scaler

def scale_with(arr: np.ndarray, scaler):
    arr2d = arr.reshape(-1, 1)
    return scaler.transform(arr2d).flatten()

def inverse_scale(arr: np.ndarray, scaler):
    arr2d = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr2d).flatten()

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
