"""
coldstart.py — Cold-start identifiability.

An all-missing column is unidentifiable in scale without side information.
In the main pipeline, the MinMax scaler for the held-out column is fitted
on that column's own observed values, i.e. the recovered values are placed
on an *oracle scale* the method would not know at true cold-start. This
script makes that explicit and quantifies the gap:

  ORACLE  : held-out column scaled by its own [min,max] (the main-pipeline setting).
  DONOR   : held-out column scaled by the *USDA SR Legacy* range for the same
            nutrient (a realistic external prior; true side information).
  NAIVE   : held-out column scaled by the pooled [min,max] of all *other*
            columns' observed values (no per-column side info) -> shows the
            identifiability limit when no scale prior is available.

Only SoftImpute is evaluated (local methods cannot impute an all-missing column).
Reports leave-one-nutrient-out NRMSE under each scaling regime.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401  puts project root on sys.path
from config import NUTRIENT_COLS
from utils import NutrientPreprocessor, evaluate_imputation
from common import load_tkpi, OPTIMAL_RANK_SI, EXP_TABLES
from usda import load_usda, USDA_NUTRIENT_COLS

HERE = Path(__file__).resolve().parent


def softimpute_coldstart(X_with_nan, max_rank=5, max_iter=100, tol=0.001, cold_init=0.5):
    """SoftImpute with cold-start-aware init (identical to v4_a4)."""
    mask_obs = ~np.isnan(X_with_nan)
    X_filled = X_with_nan.copy()
    for j in range(X_filled.shape[1]):
        col_missing = ~mask_obs[:, j]
        if not col_missing.any():
            continue
        if mask_obs[:, j].any():
            fill_val = float(np.nanmean(X_filled[:, j]))
        else:
            fill_val = cold_init
        X_filled[col_missing, j] = fill_val
    _, s0, _ = np.linalg.svd(X_filled, full_matrices=False)
    shrinkage = s0[0] / 50.0
    X_observed = np.where(mask_obs, X_with_nan, 0.0)
    for _ in range(max_iter):
        X_prev = X_filled.copy()
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        rank = min(max_rank, len(s))
        s_thresh = np.maximum(s[:rank] - shrinkage, 0.0)
        X_recon = U[:, :rank] @ np.diag(s_thresh) @ Vt[:rank, :]
        X_filled = np.where(mask_obs, X_observed, X_recon)
        change = np.linalg.norm(X_filled - X_prev) / (np.linalg.norm(X_prev) + 1e-10)
        if change < tol:
            break
    return X_filled


def usda_ranges():
    """Per-nutrient (min,max) of observed USDA values, as a donor scale prior."""
    Xu, _, ucols = load_usda()
    rng = {}
    for j, c in enumerate(ucols):
        col = Xu[:, j]
        col = col[~np.isnan(col)]
        rng[c] = (float(np.min(col)), float(np.max(col)))
    return rng


def run_regime(X_raw, col_idx, nutrient_cols, regime, donor_rng):
    """
    Fit scaler for the non-held-out columns on their observed values (leak-safe).
    For the held-out column j, choose the [min,max] used to place recovered
    values back on scale according to `regime`.
    """
    n, p = X_raw.shape
    j = col_idx
    cname = nutrient_cols[j]

    mask_test = np.zeros(X_raw.shape, dtype=bool)
    mask_test[:, j] = ~np.isnan(X_raw[:, j])
    mask_train = (~np.isnan(X_raw)) & (~mask_test)

    # Scaler for all columns fitted on their own observed values EXCEPT column j,
    # whose scale is determined by the regime.
    prep = NutrientPreprocessor(col_methods=['none'] * p)
    # fit on training-observed entries only (column j has none -> handled below)
    prep.fit(X_raw, mask_train)

    # Determine held-out column scale [lo, hi]
    obs_j = X_raw[~np.isnan(X_raw[:, j]), j]
    if regime == 'oracle':
        lo, hi = float(obs_j.min()), float(obs_j.max())
    elif regime == 'donor':
        lo, hi = donor_rng.get(cname, (float(obs_j.min()), float(obs_j.max())))
    elif regime == 'naive':
        # pooled observed range of all OTHER columns
        other = X_raw[:, [k for k in range(p) if k != j]]
        other = other[~np.isnan(other)]
        lo, hi = float(other.min()), float(other.max())
    else:
        raise ValueError(regime)

    # Overwrite the scaler for column j with the chosen [lo,hi]
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    sc.fit(np.array([[lo], [hi]]))
    prep.scalers_[j] = sc
    prep.transformers_[j] = None

    X_norm = prep.transform(X_raw)      # column j is all-NaN in X_norm
    X_input = X_norm.copy()
    X_input[mask_test] = np.nan          # (already NaN, explicit)
    X_input[:, j] = np.nan               # ensure entirely unobserved

    X_imp_norm = softimpute_coldstart(X_input, max_rank=OPTIMAL_RANK_SI)
    X_imp = prep.inverse_transform(X_imp_norm)
    res = evaluate_imputation(X_raw, X_imp, mask_test, nutrient_cols)
    return res['per_nutrient'].get(cname, {}).get('nrmse', np.nan)


if __name__ == "__main__":
    X_raw, _, nutrient_cols = load_tkpi()
    donor_rng = usda_ranges()
    # Total_Carotenoids has no USDA donor -> donor falls back to oracle for it.

    regimes = ['oracle', 'donor', 'naive']
    rows = []
    for j, c in enumerate(nutrient_cols):
        rec = {'nutrient': c, 'obs_frac': round((~np.isnan(X_raw[:, j])).mean(), 3),
               'has_usda_donor': c in donor_rng}
        for reg in regimes:
            rec[reg] = run_regime(X_raw, j, nutrient_cols, reg, donor_rng)
        rows.append(rec)
        print(f"  {c:<18} oracle={rec['oracle']:.3f}  donor={rec['donor']:.3f}  "
              f"naive={rec['naive']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(EXP_TABLES / "coldstart_scaling_regimes.csv", index=False)
    print("\nMedian NRMSE across 19 nutrients:")
    for reg in regimes:
        print(f"  {reg:<8} median={df[reg].median():.4f}  "
              f"#<1.0={(df[reg] < 1.0).sum()}")
    print(f"\nSaved -> {EXP_TABLES / 'coldstart_scaling_regimes.csv'}")
