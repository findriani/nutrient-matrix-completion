"""
mae.py — Recompute Scenario A MAE with the paper's aggregation scheme.

The paper's main-results table reports MAE as the per-seed *median* of
per-nutrient MAE (robust to large-scale nutrients like Sodium/Retinol), aggregated
as mean +/- std over seeds — the same scheme as its NRMSE column. main_tkpi.py
instead stored the per-seed *mean* of per-nutrient MAE, which is scale-dominated
(~170-220). This script re-runs Scenario A (20%, 10 seeds) for every method and
records BOTH per-seed median NRMSE (to re-validate against main_scenarioA) and
per-seed median MAE (the corrected column).

Output: outputs/tables/main_scenarioA_mae_10seed.csv  (seed x method: median_nrmse, median_mae)
        prints mean +/- std over seeds.
"""
import time
import numpy as np
import pandas as pd

from common import load_tkpi, make_random_mask, impute, SEEDS_10, EXP_TABLES
from utils import evaluate_imputation

METHODS = ['knn_k10', 'knn_k5', 'knn_k3', 'softimpute', 'masked_nmf',
           'mice', 'mice_extratrees', 'missforest', 'mean', 'median']

X_raw, _, nutrient_cols = load_tkpi()

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for seed in SEEDS_10:
        mtr, mte = make_random_mask(X_raw, 0.20, seed)
        for m in METHODS:
            X_imp = impute(m, X_raw, mtr, mte, seed=seed)
            per = evaluate_imputation(X_raw, X_imp, mte, nutrient_cols)['per_nutrient']
            nrmse = [per[c]['nrmse'] for c in nutrient_cols if c in per]
            mae = [per[c]['mae'] for c in nutrient_cols if c in per]
            rows.append(dict(seed=seed, method=m,
                             median_nrmse=float(np.median(nrmse)),
                             median_mae=float(np.median(mae))))
        pd.DataFrame(rows).to_csv(EXP_TABLES / "main_scenarioA_mae_10seed.csv", index=False)
        print(f"  seed {seed} done  [{(time.time()-t0)/60:.1f}m]", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby('method').agg(
        nrmse=('median_nrmse', 'mean'), nrmse_s=('median_nrmse', 'std'),
        mae=('median_mae', 'mean'), mae_s=('median_mae', 'std')).sort_values('nrmse')
    print("\nScenario A (20%, 10 seeds) — median-aggregated NRMSE and MAE:")
    for m, r in g.iterrows():
        print(f"  {m:<16} NRMSE={r.nrmse:.4f}+/-{r.nrmse_s:.4f}   "
              f"MAE={r.mae:.2f}+/-{r.mae_s:.2f}")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
