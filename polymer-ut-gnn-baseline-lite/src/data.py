
import numpy as np

def make_volume(grid=28, noise=0.10, seed=0):
    # Synthetic 3D amplitude volume and ground-truth IW map
    rng = np.random.RandomState(seed)
    g = grid
    ax = np.linspace(-1,1,g)
    X,Y,Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    center_r = 0.5
    width = 0.18
    iw = np.exp(-((R-center_r)**2)/(2*width*width))
    iw = (iw - iw.min()) / (iw.max() - iw.min())
    amp = np.sqrt(iw) * 0.7 + (iw**2)*0.3
    # cheap blur
    for _ in range(2):
        amp = (amp + np.roll(amp,1,0) + np.roll(amp,-1,0) + np.roll(amp,1,1) + np.roll(amp,-1,1) + np.roll(amp,1,2) + np.roll(amp,-1,2)) / 7.0
    amp = amp + rng.normal(0, noise, size=amp.shape)
    atten = 1.0 - 0.25*(Z+1.0)/2.0
    vol = amp * atten
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    return vol.astype(np.float32), iw.astype(np.float32)
