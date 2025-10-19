
import numpy as np

def volume_to_features(vol, iw):
    g = vol.shape[0]
    step = 2
    s = slice(0, g - (g % step), step)
    v = vol[s,s,s]
    t = iw[s,s,s]
    def shift(a, d, axis): return np.roll(a, d, axis=axis)
    mean6 = (v + shift(v,1,0) + shift(v,-1,0) + shift(v,1,1) + shift(v,-1,1) + shift(v,1,2) + shift(v,-1,2)) / 7.0
    var6 = ((v-mean6)**2 + (shift(v,1,0)-mean6)**2 + (shift(v,-1,0)-mean6)**2 + (shift(v,1,1)-mean6)**2 + (shift(v,-1,1)-mean6)**2 + (shift(v,1,2)-mean6)**2 + (shift(v,-1,2)-mean6)**2) / 7.0
    dx = (shift(v,1,0) - shift(v,-1,0)) * 0.5
    dy = (shift(v,1,1) - shift(v,-1,1)) * 0.5
    dz = (shift(v,1,2) - shift(v,-1,2)) * 0.5
    g2 = v.shape[0]
    ax = np.linspace(-1,1,g2)
    X,Y,Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    feats = np.stack([v, mean6, var6, np.abs(dx), np.abs(dy), np.abs(dz), R], axis=-1)
    X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    y = t.reshape(-1).astype(np.float32)
    return X, y
