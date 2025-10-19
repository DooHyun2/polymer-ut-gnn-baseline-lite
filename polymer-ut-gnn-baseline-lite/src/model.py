
import numpy as np

def fit_linear_ridge(X, y, lam=1e-6):
    Xb = np.concatenate([X, np.ones((X.shape[0],1), dtype=X.dtype)], axis=1)
    XtX = Xb.T @ Xb
    lamI = np.eye(XtX.shape[0], dtype=XtX.dtype) * lam
    w = np.linalg.pinv(XtX + lamI) @ (Xb.T @ y)
    def predict(Xnew):
        Xnb = np.concatenate([Xnew, np.ones((Xnew.shape[0],1), dtype=Xnew.dtype)], axis=1)
        return Xnb @ w
    return w, predict

def kfold_linear_baseline(X, y, k=5, lam=1e-6, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    preds = np.zeros_like(y, dtype=np.float32)
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j!=i])
        w, pred = fit_linear_ridge(X[tr_idx], y[tr_idx], lam=lam)
        preds[val_idx] = pred(X[val_idx])
    return preds, y.copy()
