
import argparse, os, json, numpy as np
from src.data import make_volume
from src.features import volume_to_features
from src.model import kfold_linear_baseline
from src.metrics import mae, pearson_r, coeff_var
from src.utils import ensure_dir, ascii_scatter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=28, help="grid size per axis (downsampled)")
    p.add_argument("--noise", type=float, default=0.10, help="noise level for UT amplitude")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3, help="repeat k-fold to estimate CV")
    p.add_argument("--folds", type=int, default=5, help="k-fold")
    args = p.parse_args()
    np.random.seed(args.seed)

    # 1) synthetic volume + ground-truth intermediate-water index (IW)
    vol, iw = make_volume(grid=args.grid, noise=args.noise, seed=args.seed)

    # 2) simple graph-like features per voxel
    X, y = volume_to_features(vol, iw)

    # 3) linear baseline with k-fold, repeated to estimate variability
    preds_list, trues_list = [], []
    r_list, mae_list = [], []
    for rep in range(args.repeats):
        pr, tr = kfold_linear_baseline(X, y, k=args.folds, seed=args.seed+rep)
        preds_list.append(pr); trues_list.append(tr)
        r_list.append(pearson_r(tr, pr))
        mae_list.append(mae(tr, pr))

    # aggregate
    r_mean, r_std = float(np.mean(r_list)), float(np.std(r_list, ddof=1) if len(r_list)>1 else 0.0)
    mae_mean, mae_std = float(np.mean(mae_list)), float(np.std(mae_list, ddof=1) if len(mae_list)>1 else 0.0)
    cv = coeff_var(np.array(mae_list))

    # save artifacts
    out_dir = "artifacts"
    ensure_dir(out_dir)
    # save metrics
    metrics = {"r_mean": r_mean, "r_std": r_std, "mae_mean": mae_mean, "mae_std": mae_std, "cv_mae": cv}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # save sample predictions
    import csv
    pr = preds_list[0]; tr = trues_list[0]
    n_dump = min(5000, len(pr))
    with open(os.path.join(out_dir, "pred_vs_true.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true","pred"])
        for i in range(n_dump):
            w.writerow([float(tr[i]), float(pr[i])])
    # ascii scatter for 1-page plan
    scat = ascii_scatter(tr[:n_dump], pr[:n_dump], bins=24)
    with open(os.path.join(out_dir, "fig_pred_vs_true.txt"), "w") as f:
        f.write(scat)
    # textual summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Pearson r = {r_mean:.3f} (±{r_std:.3f}), MAE = {mae_mean:.3f} (±{mae_std:.3f}), CV(MAE)={cv:.3f}\n")
        f.write("This is a minimal end-to-end baseline. Replace with GNN/3D-UNet later.\n")

    print("Done. Metrics:", metrics)
    print("Artifacts saved to ./artifacts")

if __name__ == "__main__":
    main()
