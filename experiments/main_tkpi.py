"""
main_tkpi.py — 10-seed TKPI Scenario A & B main results.

For each seed:
  Scenario A = 20% random mask (make_random_mask).
  Scenario B = block-micro mask (30% of foods lose all micronutrients).
Computes per-nutrient NRMSE + MAE for every method, aggregates to per-seed
median (NRMSE) and per-seed mean MAE, then reports mean +/- std over 10 seeds.
Also runs the Wilcoxon signed-rank test (KNN k=10 vs SoftImpute) with n=10.

Outputs:
  outputs/tables/main_scenarioA_10seed.csv     (per seed x method)
  outputs/tables/main_scenarioB_10seed.csv
  outputs/tables/per_nutrient_A_10seed.csv      (per nutrient, mean over seeds)
  outputs/tables/wilcoxon_10seed.txt
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from _bootstrap import PROJECT_ROOT  # noqa: F401  puts project root on sys.path
from config import MICRO_COLS, NUTRIENT_COLS
from utils import evaluate_imputation
from common import (
    load_tkpi, make_random_mask, impute, SEEDS_10, EXP_TABLES,
)

METHODS = ['knn_k10', 'knn_k5', 'knn_k3', 'softimpute', 'masked_nmf',
           'mice', 'mice_extratrees', 'missforest', 'mean', 'median']

X_raw, food_ids, nutrient_cols = load_tkpi()
micro_idx = [nutrient_cols.index(c) for c in MICRO_COLS]


def block_micro_mask(X_raw, food_frac, seed):
    """Scenario B: remove all micronutrient values for a random food_frac of foods."""
    rng = np.random.default_rng(seed)
    n = X_raw.shape[0]
    n_foods = int(n * food_frac)
    chosen = rng.choice(n, size=n_foods, replace=False)
    mask_test = np.zeros(X_raw.shape, dtype=bool)
    for i in chosen:
        for j in micro_idx:
            if not np.isnan(X_raw[i, j]):
                mask_test[i, j] = True
    mask_train = (~np.isnan(X_raw)) & (~mask_test)
    return mask_train, mask_test


def run_scenario(mask_fn, label):
    rowsA = []
    per_nut_accum = {m: {c: [] for c in nutrient_cols} for m in METHODS}
    for seed in SEEDS_10:
        mtr, mte = mask_fn(X_raw, seed)
        for m in METHODS:
            X_imp = impute(m, X_raw, mtr, mte, seed=seed)
            res = evaluate_imputation(X_raw, X_imp, mte, nutrient_cols)
            per = res['per_nutrient']
            nrmse_list = [per[c]['nrmse'] for c in nutrient_cols if c in per]
            mae_list = [per[c]['mae'] for c in nutrient_cols if c in per]
            rowsA.append(dict(seed=seed, method=m,
                              median_nrmse=float(np.median(nrmse_list)),
                              mean_mae=float(np.mean(mae_list))))
            for c in nutrient_cols:
                if c in per:
                    per_nut_accum[m][c].append(per[c]['nrmse'])
        print(f"  [{label}] seed {seed} done", flush=True)
    df = pd.DataFrame(rowsA)
    return df, per_nut_accum


def summarise(df, label):
    print(f"\n=== Scenario {label}: mean +/- std over {df.seed.nunique()} seeds ===")
    g = df.groupby('method').agg(
        nrmse_mean=('median_nrmse', 'mean'), nrmse_std=('median_nrmse', 'std'),
        mae_mean=('mean_mae', 'mean'), mae_std=('mean_mae', 'std'))
    g = g.sort_values('nrmse_mean')
    for m, r in g.iterrows():
        print(f"  {m:<16} NRMSE={r.nrmse_mean:.4f}+/-{r.nrmse_std:.4f}   "
              f"MAE={r.mae_mean:.2f}+/-{r.mae_std:.2f}")
    return g


if __name__ == "__main__":
    t0 = time.time()

    dfA, per_nut_A = run_scenario(lambda X, s: make_random_mask(X, 0.20, s), "A")
    dfA.to_csv(EXP_TABLES / "main_scenarioA_10seed.csv", index=False)
    gA = summarise(dfA, "A (20% random)")

    dfB, _ = run_scenario(lambda X, s: block_micro_mask(X, 0.30, s), "B")
    dfB.to_csv(EXP_TABLES / "main_scenarioB_10seed.csv", index=False)
    gB = summarise(dfB, "B (block micro)")

    # per-nutrient table (Scenario A, mean over seeds) for SoftImpute vs best baseline
    pn_rows = []
    for c in nutrient_cols:
        rec = {'nutrient': c}
        for m in METHODS:
            vals = per_nut_A[m][c]
            rec[m] = float(np.mean(vals)) if vals else np.nan
        pn_rows.append(rec)
    pd.DataFrame(pn_rows).to_csv(EXP_TABLES / "per_nutrient_A_10seed.csv", index=False)

    # Wilcoxon KNN k=10 vs SoftImpute across 10 seeds (Scenario A)
    knn = dfA[dfA.method == 'knn_k10'].sort_values('seed').median_nrmse.values
    si = dfA[dfA.method == 'softimpute'].sort_values('seed').median_nrmse.values
    stat, p = wilcoxon(knn, si)
    txt = (f"Wilcoxon signed-rank (Scenario A, n={len(knn)} seeds)\n"
           f"KNN k=10 median NRMSE per seed: {np.round(knn,4).tolist()}\n"
           f"SoftImpute median NRMSE per seed: {np.round(si,4).tolist()}\n"
           f"mean KNN={knn.mean():.4f}  mean SI={si.mean():.4f}\n"
           f"W={stat:.3f}  p={p:.4f}\n")
    (EXP_TABLES / "wilcoxon_10seed.txt").write_text(txt)
    print("\n" + txt)
    print(f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)
