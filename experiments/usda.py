"""
usda.py — USDA SR Legacy loader + missingness-mechanism mask generators.

Loads the prebuilt wide matrix (usda_sr_legacy_2018_wide.csv, 18 nutrients) and
provides MCAR, MAR, and MNAR maskers over the naturally observed entries, so the
crossover sweep can be repeated under each mechanism.

Mechanism definitions (all hold out ~`frac` of naturally observed entries):
  MCAR : uniform random over observed entries.
  MAR  : per-cell missingness probability increases with a *fully observed*
         driver column (Energy) — missingness depends only on observed data.
  MNAR : low-value self-masking — within each column the smallest values are the
         most likely to be held out (models limit-of-detection censoring of
         trace nutrients). Depends on the value that goes missing.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from _bootstrap import PROJECT_ROOT

HERE = PROJECT_ROOT   # raw data files live at the project root

USDA_NUTRIENT_COLS = [
    'Energy', 'Protein', 'Fat', 'Carbohydrate', 'Fiber',
    'Calcium', 'Phosphorus', 'Iron', 'Sodium', 'Potassium',
    'Copper', 'Zinc', 'Retinol', 'Beta_Carotene',
    'Thiamine', 'Riboflavin', 'Niacin', 'Vitamin_C',
]
USDA_TRANSFORM_MAP = ['none'] * len(USDA_NUTRIENT_COLS)


def load_usda():
    """Return (X_raw [n,18] float w/ NaN, food_ids, cols) from the wide CSV."""
    df = pd.read_csv(HERE / "usda_sr_legacy_2018_wide.csv")
    X = df[USDA_NUTRIENT_COLS].values.astype(float)
    food_ids = df['fdc_id'].values
    return X, food_ids, list(USDA_NUTRIENT_COLS)


def _target_count(X, frac):
    observed = np.argwhere(~np.isnan(X))
    return observed, int(len(observed) * frac)


def make_mcar_mask(X, frac, seed):
    rng = np.random.default_rng(seed)
    observed, n_hold = _target_count(X, frac)
    chosen = rng.choice(len(observed), size=n_hold, replace=False)
    mask_test = np.zeros(X.shape, dtype=bool)
    for idx in observed[chosen]:
        mask_test[idx[0], idx[1]] = True
    mask_train = (~np.isnan(X)) & (~mask_test)
    return mask_train, mask_test


def make_mar_mask(X, frac, seed, driver_col=0):
    """
    MAR: hold-out probability rises with an observed driver column (Energy, col 0),
    which is 100% observed in SR Legacy so selection depends only on observed data.
    Sampling is weighted by the row's driver rank; the driver column itself is
    never held out (it is the observed conditioning variable).
    """
    rng = np.random.default_rng(seed)
    driver = X[:, driver_col]
    # rank-normalise driver to [0.1, 1.0] weight per row (higher energy -> more missing)
    order = np.argsort(np.argsort(driver))
    row_w = 0.1 + 0.9 * (order / (len(order) - 1))

    observed = np.argwhere(~np.isnan(X))
    # exclude driver column from being masked
    keep = observed[:, 1] != driver_col
    observed = observed[keep]
    w = row_w[observed[:, 0]]
    w = w / w.sum()
    n_hold = int(len(observed) * frac)
    chosen = rng.choice(len(observed), size=n_hold, replace=False, p=w)
    mask_test = np.zeros(X.shape, dtype=bool)
    for idx in observed[chosen]:
        mask_test[idx[0], idx[1]] = True
    mask_train = (~np.isnan(X)) & (~mask_test)
    return mask_train, mask_test


def make_mnar_mask(X, frac, seed, steepness=6.0):
    """
    MNAR: low-value self-masking (limit-of-detection style). Within each column,
    hold-out probability decreases with the value's within-column percentile, so
    the smallest concentrations are most likely to be censored. Missingness
    depends on the value that becomes missing.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    weight = np.zeros((n, p))
    for j in range(p):
        col = X[:, j]
        obs = ~np.isnan(col)
        if obs.sum() < 2:
            continue
        pct = np.zeros(n)
        order = np.argsort(np.argsort(col[obs]))
        pct[obs] = order / (obs.sum() - 1)          # 0=smallest .. 1=largest
        # low values -> high weight
        weight[obs, j] = np.exp(-steepness * pct[obs])
    observed = np.argwhere(~np.isnan(X))
    w = weight[observed[:, 0], observed[:, 1]]
    w = w / w.sum()
    n_hold = int(len(observed) * frac)
    chosen = rng.choice(len(observed), size=n_hold, replace=False, p=w)
    mask_test = np.zeros(X.shape, dtype=bool)
    for idx in observed[chosen]:
        mask_test[idx[0], idx[1]] = True
    mask_train = (~np.isnan(X)) & (~mask_test)
    return mask_train, mask_test


MECHANISMS = {'mcar': make_mcar_mask, 'mar': make_mar_mask, 'mnar': make_mnar_mask}
