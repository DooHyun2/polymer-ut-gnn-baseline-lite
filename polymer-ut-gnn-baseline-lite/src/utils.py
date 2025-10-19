
import os, numpy as np

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def ascii_scatter(y, p, bins=24):
    y = np.asarray(y).flatten()
    p = np.asarray(p).flatten()
    y_n = (y - y.min())/(y.max()-y.min()+1e-12)
    p_n = (p - p.min())/(p.max()-p.min()+1e-12)
    grid = np.zeros((bins, bins), dtype=int)
    for yi, pi in zip(y_n, p_n):
        i = min(bins-1, max(0, int(yi*(bins-1))))
        j = min(bins-1, max(0, int(pi*(bins-1))))
        grid[bins-1-i, j] += 1
    chars = " .:-=+*#%@"
    levels = np.linspace(0, grid.max() if grid.max()>0 else 1, len(chars))
    lines = []
    for r in range(bins):
        line = ""
        for c in range(bins):
            v = grid[r,c]
            idx = np.searchsorted(levels, v, side="right")-1
            idx = int(np.clip(idx, 0, len(chars)-1))
            line += chars[idx]
        lines.append(line)
    return "\n".join(lines)
