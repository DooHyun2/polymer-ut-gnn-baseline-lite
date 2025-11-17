polymer-ut-gnn-baseline-lite
This repository is a minimal UT x ML baseline designed to demonstrate the ability to:
* handle 3D ultrasound-like inspection volumes
* extract graph-style features
* report simple baseline metrics (MAE/ r / CV)
It is intentionally lightweight for clarity and reproducibility.

Purpose
A compact "evidence-ready" pipeline for research proposals.
The goal is to demonstrate:
* preprocessing of 3D UT-like volumetric data
* graph-style feature extraction from spatial grids
* simple baseline modeling and evaluation
* reproducibility using only Python + Numpy
  This structure can be naturally extended to full GNN models or 3D-UNet pipelines after joining a laboratory.

How to Run
Verify NumPy:
python -c "import numpy; print('numpy OK')"
Run the baseline:
python run_baseline.py --seed 0 --grid 28 --noise 0.10
This generates a small synthetic 3D volume, extracts graph-style features,trains a linear baseline model, and prints MAE / r /5-fold CV metrics.

