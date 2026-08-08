"""
runtime.py — wall-clock timing for every imputation method, including ISVD/DAE/PICA.

Times each imputation method end-to-end on TKPI at 20% missing, averaged over 3
seeds. Timing covers the full impute-to-filled-matrix call on the leakage-safe
MinMax pipeline (the same path used for accuracy), so numbers are comparable
across methods. Single-threaded contention is avoided by running serially.

Output: outputs/tables/runtime.csv  (method, sec_mean, sec_std, n)
"""
import time
import numpy as np
import pandas as pd

from common import (
    load_tkpi, make_random_mask, impute, _prep_norm, EXP_TABLES,
)
from dl import dae_impute
from pica import pica_impute

# methods routed through the shared impute() dispatcher
IMPUTE_METHODS = ['mean', 'median', 'knn_k3', 'knn_k5', 'knn_k10',
                  'softimpute', 'iterativesvd', 'masked_nmf',
                  'mice', 'mice_extratrees', 'missforest']
SEEDS = [123, 456, 789]
FRAC = 0.20

X_raw, _, cols = load_tkpi()


def time_method(method, mtr, mte, seed):
    t = time.perf_counter()
    if method == 'dae':
        prep, Xin = _prep_norm(X_raw, mtr, mte)
        prep.inverse_transform(dae_impute(Xin, seed=seed))
    elif method == 'pica':
        prep, Xin = _prep_norm(X_raw, mtr, mte)
        prep.inverse_transform(pica_impute(Xin))
    else:
        impute(method, X_raw, mtr, mte, seed=seed)
    return time.perf_counter() - t


if __name__ == "__main__":
    t0 = time.time()
    rec = {m: [] for m in IMPUTE_METHODS + ['dae', 'pica']}
    for seed in SEEDS:
        mtr, mte = make_random_mask(X_raw, FRAC, seed)
        for m in rec:
            # one warm-up-free timed call per (method, seed)
            rec[m].append(time_method(m, mtr, mte, seed))
        print(f"  seed {seed} done  [{(time.time()-t0)/60:.1f}m]", flush=True)

    rows = []
    for m, ts in rec.items():
        ts = np.array(ts)
        rows.append(dict(method=m, sec_mean=ts.mean(), sec_std=ts.std(), n=len(ts)))
    df = pd.DataFrame(rows).sort_values('sec_mean')
    df.to_csv(EXP_TABLES / "runtime.csv", index=False)
    print("\nTKPI @20% wall-clock (mean of 3 seeds):")
    for _, r in df.iterrows():
        print(f"  {r.method:<16} {r.sec_mean:8.3f} s  (+/- {r.sec_std:.3f})")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
