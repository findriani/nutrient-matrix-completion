"""
common.py — Shared infrastructure for the experiments in this folder.

One authoritative, leakage-safe pipeline (10 seeds, every baseline, MAR/MNAR,
downstream tasks) that every other script here builds on, so results are
consistent across the whole suite.

Builds on config.py / utils.py for preprocessing and evaluation.

SoftImpute uses fancyimpute (compat_patch adapts it to scikit-learn >= 1.6).
"""

from _bootstrap import PROJECT_ROOT  # noqa: F401  puts project root on sys.path
import compat_patch  # noqa: F401  MUST precede fancyimpute import (patches check_array)

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from fancyimpute import SoftImpute

from config import (
    NUTRIENT_COLS, COLUMN_RENAME, ID_COL, DATA_PATH,
    NUTRIENT_TRANSFORM_MAP, TEST_SEEDS,
)
from utils import NutrientPreprocessor, evaluate_imputation

# ── Canonical settings from the paper ────────────────────────────────────────
OPTIMAL_RANK_SI  = 5
OPTIMAL_RANK_NMF = 2
NMF_SEED         = 42          # RANDOM_SEED used for NMF W/H init in main results
MISSING_FRACS    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# 10 seeds: the original 4 (123,456,789,1024) plus 6 new. Original 4 kept first
# so the 4-seed subset remains exactly reproducible.
SEEDS_4  = list(TEST_SEEDS)                       # [123, 456, 789, 1024]
SEEDS_10 = SEEDS_4 + [1337, 2024, 7, 99, 555, 31337]


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════
def load_tkpi():
    """Return (X_raw [1146,19] float with NaN for missing, food_ids, nutrient_cols)."""
    df = pd.read_csv(DATA_PATH)
    df.rename(columns=COLUMN_RENAME, inplace=True)
    X_raw = df[NUTRIENT_COLS].values.astype(float)
    food_ids = df[ID_COL].values
    return X_raw, food_ids, list(NUTRIENT_COLS)


# ═════════════════════════════════════════════════════════════════════════════
# Masking
# ═════════════════════════════════════════════════════════════════════════════
def make_random_mask(X_raw, frac, seed):
    """Scenario A / sweep: hold out `frac` of observed entries uniformly (MCAR)."""
    rng = np.random.default_rng(seed)
    observed_idx = np.argwhere(~np.isnan(X_raw))
    n_hold = int(len(observed_idx) * frac)
    chosen = rng.choice(len(observed_idx), size=n_hold, replace=False)
    mask_test = np.zeros(X_raw.shape, dtype=bool)
    for idx in observed_idx[chosen]:
        mask_test[idx[0], idx[1]] = True
    mask_train = (~np.isnan(X_raw)) & (~mask_test)
    return mask_train, mask_test


# ═════════════════════════════════════════════════════════════════════════════
# Masked NMF (exact copy of create_notebooks.py implementation)
# ═════════════════════════════════════════════════════════════════════════════
def masked_nmf(X_obs_matrix, mask_obs, rank, n_iter=300, eps=1e-10, seed=42):
    rng = np.random.default_rng(seed)
    n, p = X_obs_matrix.shape
    W = rng.uniform(0, 1, (n, rank)).astype(np.float64) + eps
    H = rng.uniform(0, 1, (rank, p)).astype(np.float64) + eps
    X = np.where(mask_obs, X_obs_matrix, 0.0).astype(np.float64)
    M = mask_obs.astype(np.float64)
    prev_loss = np.inf
    for it in range(n_iter):
        WH = W @ H
        numer_H = W.T @ (M * X)
        denom_H = W.T @ (M * WH) + eps
        H *= numer_H / denom_H
        H = np.maximum(H, eps)
        WH = W @ H
        numer_W = (M * X) @ H.T
        denom_W = (M * WH) @ H.T + eps
        W *= numer_W / denom_W
        W = np.maximum(W, eps)
        if it % 50 == 49:
            WH = W @ H
            loss = float(np.sum(M * (X - WH) ** 2))
            if abs(prev_loss - loss) / (prev_loss + eps) < 1e-6:
                break
            prev_loss = loss
    return W @ H


# ═════════════════════════════════════════════════════════════════════════════
# Unified impute-and-evaluate
# ═════════════════════════════════════════════════════════════════════════════
def _prep_norm(X_raw, mask_train, mask_test, col_methods=None):
    """Leakage-safe MinMax normalisation; returns (prep, X_input_with_nan_at_test)."""
    if col_methods is None:
        col_methods = NUTRIENT_TRANSFORM_MAP
    prep = NutrientPreprocessor(col_methods=col_methods)
    prep.fit(X_raw, mask_train)
    X_norm = prep.transform(X_raw)
    X_input = X_norm.copy()
    X_input[mask_test] = np.nan
    return prep, X_input


def impute(method, X_raw, mask_train, mask_test, seed=42, col_methods=None):
    """
    Return imputed matrix in ORIGINAL scale for the given method name.

    Supported: softimpute, knn_k3, knn_k5, knn_k10, masked_nmf, mice,
               mice_extratrees, missforest, mean, median,
               iterativesvd, nnm  (added as extra baselines).

    col_methods lets callers set the per-column transform list length to match
    the matrix width (TKPI=19, USDA=18). Defaults to TKPI's NUTRIENT_TRANSFORM_MAP.
    """
    prep, X_input = _prep_norm(X_raw, mask_train, mask_test, col_methods=col_methods)

    if method == 'softimpute':
        X_imp_norm = SoftImpute(max_rank=OPTIMAL_RANK_SI, verbose=False).fit_transform(X_input)

    elif method.startswith('knn_k'):
        k = int(method.rsplit('_k', 1)[1])
        X_imp_norm = KNNImputer(n_neighbors=k, weights='distance',
                                keep_empty_features=True).fit_transform(X_input)

    elif method == 'masked_nmf':
        mask_obs = ~np.isnan(X_input)
        X_for_nmf = np.maximum(np.where(mask_obs, X_input, 0.0), 0.0)
        X_imp_norm = masked_nmf(X_for_nmf, mask_obs, rank=OPTIMAL_RANK_NMF, seed=seed)

    elif method == 'mice':
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               min_value=0, random_state=seed)
        X_imp_norm = imp.fit_transform(X_input)

    elif method == 'mice_extratrees':
        imp = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            max_iter=10, min_value=0, random_state=seed)
        X_imp_norm = imp.fit_transform(X_input)

    elif method == 'missforest':
        # n_jobs=-1 parallelises the forest across cores; with random_state fixed
        # the imputed values are identical to the single-core run, only faster.
        imp = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            max_iter=10, min_value=0, random_state=seed)
        X_imp_norm = imp.fit_transform(X_input)

    elif method == 'mean':
        X_imp_norm = SimpleImputer(strategy='mean').fit_transform(X_input)

    elif method == 'median':
        X_imp_norm = SimpleImputer(strategy='median').fit_transform(X_input)

    elif method == 'iterativesvd':
        from fancyimpute import IterativeSVD
        X_imp_norm = IterativeSVD(rank=OPTIMAL_RANK_SI, verbose=False).fit_transform(X_input)

    elif method == 'nnm':
        from fancyimpute import NuclearNormMinimization
        X_imp_norm = NuclearNormMinimization(verbose=False).fit_transform(X_input)

    else:
        raise ValueError(f"Unknown method: {method}")

    return prep.inverse_transform(X_imp_norm)


def impute_and_nrmse(method, X_raw, mask_train, mask_test, nutrient_cols, seed=42,
                     col_methods=None):
    X_imp = impute(method, X_raw, mask_train, mask_test, seed=seed, col_methods=col_methods)
    return evaluate_imputation(X_raw, X_imp, mask_test, nutrient_cols)['median_nrmse']


# ── Output dirs ──────────────────────────────────────────────────────────────
EXP_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EXP_TABLES = EXP_OUTPUT_DIR / "tables"
EXP_FIGS = EXP_OUTPUT_DIR / "figures"
for _d in (EXP_TABLES, EXP_FIGS):
    _d.mkdir(parents=True, exist_ok=True)
