"""
sweep_usda.py — 10-seed USDA SR Legacy MCAR sweep (5%-50%).

Methods: KNN k=10, KNN k=5, SoftImpute (matches the paper's USDA generalisation,
which excludes MissForest for cost). Extends the original 3-seed run to 10 seeds.
"""
import time
import numpy as np
import pandas as pd

from usda import load_usda, make_mcar_mask, USDA_TRANSFORM_MAP
from common import impute_and_nrmse, MISSING_FRACS, SEEDS_10, EXP_TABLES

OUT = EXP_TABLES / "sweep_usda_10seed.csv"
METHODS = ['knn_k10', 'knn_k5', 'softimpute']

X_raw, _, cols = load_usda()
print(f"USDA {X_raw.shape}  seeds={SEEDS_10}", flush=True)

rows = []
t0 = time.time()
for frac in MISSING_FRACS:
    for seed in SEEDS_10:
        mtr, mte = make_mcar_mask(X_raw, frac, seed)
        for m in METHODS:
            nrmse = impute_and_nrmse(m, X_raw, mtr, mte, cols, seed=seed,
                                     col_methods=USDA_TRANSFORM_MAP)
            rows.append(dict(missing_frac=frac, seed=seed, method=m, nrmse=nrmse))
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    sub = df[df.missing_frac == frac]
    means = {m: sub[sub.method == m].nrmse.mean() for m in METHODS}
    best = min(means, key=means.get)
    print(f"[{int(frac*100):3d}%] " +
          "  ".join(f"{m}={means[m]:.4f}" for m in METHODS) +
          f"   BEST={best}  [{(time.time()-t0)/60:.1f}m]", flush=True)

print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
