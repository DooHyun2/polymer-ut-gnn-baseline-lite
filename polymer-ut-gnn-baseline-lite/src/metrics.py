
import numpy as np

def mae(y, p):
    return float(np.mean(np.abs(y - p)))

def pearson_r(y, p):
    y = y - y.mean()
    p = p - p.mean()
    num = float((y * p).sum())
    den = float(np.sqrt((y**2).sum()) * np.sqrt((p**2).sum()))
    return (num / den) if den != 0 else 0.0

def coeff_var(arr):
    arr = np.array(arr, dtype=float)
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if len(arr)>1 else 0.0
    return (s / m) if m!=0 else 0.0
