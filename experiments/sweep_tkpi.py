"""
sweep_tkpi.py — 10-seed TKPI missing-rate sweep (5%-50%).

Methods: KNN k=10, KNN k=5, SoftImpute, Masked NMF, MissForest.
Writes tidy rows incrementally to outputs/tables/sweep_tkpi_10seed.csv
so progress is observable while MissForest (the slow method) runs.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from common import (
    load_tkpi, make_random_mask, impute_and_nrmse,
    MISSING_FRACS, SEEDS_10, EXP_TABLES,
)

OUT = EXP_TABLES / "sweep_tkpi_10seed.csv"
METHODS = ['knn_k10', 'knn_k5', 'softimpute', 'masked_nmf', 'missforest']

X_raw, _, nutrient_cols = load_tkpi()
print(f"TKPI {X_raw.shape}  seeds={SEEDS_10}", flush=True)

# Resume: keep any (frac,seed,method) rows already computed in a prior run.
rows = []
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    rows = prev.to_dict('records')
    done = {(round(r['missing_frac'], 3), int(r['seed']), r['method']) for r in rows}
    print(f"Resuming: {len(done)} cells already done", flush=True)

t0 = time.time()
for frac in MISSING_FRACS:
    for seed in SEEDS_10:
        mtr, mte = make_random_mask(X_raw, frac, seed)
        for m in METHODS:
            if (round(frac, 3), seed, m) in done:
                continue
            ts = time.time()
            nrmse = impute_and_nrmse(m, X_raw, mtr, mte, nutrient_cols, seed=seed)
            rows.append(dict(missing_frac=frac, seed=seed, method=m, nrmse=nrmse))
            if m == 'missforest':
                dt = time.time() - ts
                print(f"  frac={int(frac*100)}% seed={seed} missforest={nrmse:.4f} "
                      f"({dt:.0f}s)  [elapsed {(time.time()-t0)/60:.1f}m]", flush=True)
        # flush after each seed
        pd.DataFrame(rows).to_csv(OUT, index=False)
    # per-rate mean summary
    df = pd.DataFrame(rows)
    sub = df[df.missing_frac == frac]
    means = {m: sub[sub.method == m].nrmse.mean() for m in METHODS}
    best = min(means, key=means.get)
    print(f"[{int(frac*100):3d}%] " +
          "  ".join(f"{m}={means[m]:.4f}" for m in METHODS) +
          f"   BEST={best}", flush=True)

print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
