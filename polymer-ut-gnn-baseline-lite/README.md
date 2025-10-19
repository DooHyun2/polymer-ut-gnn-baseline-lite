
# polymer-ut-gnn-baseline-lite

A tiny **evidence** repo to attach to a research plan.
It simulates 3D ultrasound-like volumes for a polymer film, creates simple **graph-like node features**,
fits a **linear baseline** (closed-form) to predict a synthetic "intermediate water index",
and reports **MAE / Pearson r / CV**. No external dependencies except `numpy`.

> This is *not* a real GNN/3D-UNet. It is a lightweight **baseline** to demonstrate the end-to-end pipeline
> without heavy frameworks. It can be replaced with PyTorch later.

## How to run

```bash
# 1) Ensure Python 3.8+ and numpy are available
python -c "import numpy; print('numpy', numpy.__version__)"

# 2) Run baseline (generates data, trains, evaluates, saves artifacts into ./artifacts)
python run_baseline.py --seed 0 --grid 28 --noise 0.10
```

Outputs:
- `artifacts/metrics.json` : MAE, r, CV (coefficient of variation across repeats)
- `artifacts/pred_vs_true.csv` : first 5k node-level predictions vs ground-truth
- `artifacts/fig_pred_vs_true.txt` : tiny ASCII scatter "thumbnail" (for 1-page plan evidence box)
- `artifacts/summary.txt` : short textual summary for pasting into a PDF

## Files
- `run_baseline.py` : orchestrates data, features, model, metrics
- `src/data.py` : synthetic 3D volume generator (ultrasound-like amplitude + ground-truth IW map)
- `src/features.py` : voxel → graph-like node features (local mean/var, gradients, radius etc.)
- `src/model.py` : closed-form linear regression (ridge=0), simple k-fold repeat
- `src/metrics.py` : MAE, Pearson r, CV
- `src/utils.py` : helpers (downsampling, neighborhoods, ascii scatter)

## Suggested citation line (for your 1-page plan)
> We provide a minimal reproducibility baseline (no external deps) achieving **r ≈ 0.80** and **MAE ≈ 0.05** on synthetic UT-like volumes (n=1–3), with weekly device usage under 2h (offline analysis prioritized).

---

**NOTE:** Replace this baseline with a proper GNN/3D-UNet when lab resources allow.
