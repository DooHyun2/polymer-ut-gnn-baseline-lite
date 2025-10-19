# polymer-ut-gnn-baseline-lite

초소형 **증거용** 파이프라인: 3D UT-유사 볼륨 → 그래프풍 특징 → 선형 베이스라인 → **MAE / r / CV** 리포트.  
의존성: **Python + numpy** (추가 설치 불필요)

## Why
- 연구계획서의 **Evidence Box**를 위한 최소 재현 파이프라인
- 입실 후 **GNN/3D-UNet**으로 교체 가능한 구조

## How to run
```bash
python -c "import numpy; print('numpy OK')"
python run_baseline.py --seed 0 --grid 28 --noise 0.10
