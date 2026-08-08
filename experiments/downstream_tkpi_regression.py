"""
downstream_tkpi_regression.py — Leave-one-nutrient-out regression on TKPI
with natural missingness.

For each target nutrient, use the other 18 as features. Only rows where the
target is observed are used (ground-truth target, imputed features). Three
regressor families to avoid imputer–regressor coupling bias.

Targets:
  Macro:  Protein (n≈1145), Fat (n≈1145), Carbohydrate (n≈1146)
  Micro:  Calcium (n≈1106), Iron (n≈1103), Vitamin_C (n≈857)

Regressors:
  RF    — RandomForestRegressor (200 trees)
  Ridge — Ridge regression (StandardScaler + alpha=1.0)
  KNR   — KNeighborsRegressor (k=5, StandardScaler)

Metric: R² (coefficient of determination) via 5-fold CV, 5 random seeds.

Output: downstream_tkpi_regression.csv        (per-seed, per-target, per-clf)
        downstream_tkpi_regression_summary.csv (method × target × clf summary)
"""
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold

from common import (load_tkpi, impute, _prep_norm,
                        EXP_TABLES, NUTRIENT_TRANSFORM_MAP)
from pica import pica_impute
from _bootstrap import PROJECT_ROOT

CV_SEEDS = [0, 1, 2, 3, 4]
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']
COL_METHODS = NUTRIENT_TRANSFORM_MAP

TARGETS = {
    'Protein':      'macro',
    'Fat':          'macro',
    'Carbohydrate': 'macro',
    'Calcium':      'micro',
    'Iron':         'micro',
    'Vitamin_C':    'micro',
}

REGRESSORS = ['rf', 'ridge', 'knr']


def load_all():
    X, ids, cols = load_tkpi()
    # drop singleton Q group (same as classification scripts)
    groups = np.array([k[0] for k in ids])
    keep = groups != 'Q'
    return X[keep], cols


def impute_full(method, X_raw):
    mask_train = ~np.isnan(X_raw)
    mask_test = np.zeros(X_raw.shape, dtype=bool)
    if method == 'pica':
        prep, Xin = _prep_norm(X_raw, mask_train, mask_test, col_methods=COL_METHODS)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, X_raw, mask_train, mask_test, seed=42,
                  col_methods=COL_METHODS)


def make_regressor(reg_type, seed):
    if reg_type == 'rf':
        return RandomForestRegressor(n_estimators=200, random_state=seed,
                                     n_jobs=-1)
    elif reg_type == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
    elif reg_type == 'knr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knr', KNeighborsRegressor(n_neighbors=5))
        ])


def reg_scores(X_feat, y_target, reg_type, seed):
    reg = make_regressor(reg_type, seed)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r = cross_validate(reg, X_feat, y_target, cv=cv,
                       scoring=['r2', 'neg_root_mean_squared_error'])
    return (float(r['test_r2'].mean()),
            float(-r['test_neg_root_mean_squared_error'].mean()))


if __name__ == "__main__":
    t0 = time.time()
    X_raw, cols = load_all()
    col_idx = {c: i for i, c in enumerate(cols)}
    print(f"TKPI: {X_raw.shape}, targets: {list(TARGETS.keys())}", flush=True)

    # Show per-target observation counts
    for tgt in TARGETS:
        n_obs = int(np.sum(~np.isnan(X_raw[:, col_idx[tgt]])))
        print(f"  {tgt:15s}: {n_obs} observed rows", flush=True)

    # Resume support
    csv_path = EXP_TABLES / "downstream_tkpi_regression.csv"
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["method"], r["target"], r["regressor"], int(r["cv_seed"]))
                for r in rows}
        print(f"Resuming: {len(done)} cells already done", flush=True)
    else:
        rows, done = [], set()

    for m in METHODS:
        ts = time.time()
        print(f"\n  Imputing {m}...", end="", flush=True)
        X_imp = impute_full(m, X_raw)
        print(f" done ({time.time()-ts:.0f}s)", flush=True)

        for tgt, tgt_type in TARGETS.items():
            ti = col_idx[tgt]
            # rows where target is observed (ground truth)
            obs_mask = ~np.isnan(X_raw[:, ti])
            y = X_raw[obs_mask, ti]
            # features = all other columns from the imputed matrix
            feat_idx = [i for i in range(len(cols)) if i != ti]
            X_feat = X_imp[obs_mask][:, feat_idx]

            for rt in REGRESSORS:
                for cv_seed in CV_SEEDS:
                    if (m, tgt, rt, cv_seed) in done:
                        continue
                    r2, rmse = reg_scores(X_feat, y, rt, cv_seed)
                    rows.append(dict(method=m, target=tgt, target_type=tgt_type,
                                     regressor=rt, cv_seed=cv_seed,
                                     r2=r2, rmse=rmse))
                    pd.DataFrame(rows).to_csv(csv_path, index=False)

                r2s = [r['r2'] for r in rows
                       if r['method'] == m and r['target'] == tgt
                       and r['regressor'] == rt]
                print(f"    {tgt:15s} {rt:5s}: R²={np.mean(r2s):.4f}", flush=True)

        print(f"    [{(time.time()-t0)/60:.1f}m total]", flush=True)

    # Summary: mean R² per method × target × regressor
    df = pd.DataFrame(rows)
    g = df.groupby(['method', 'target', 'regressor']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        rmse_mean=('rmse', 'mean')).reset_index()
    g.to_csv(EXP_TABLES / "downstream_tkpi_regression_detail.csv", index=False)

    # Compact summary: mean R² across all 6 targets per method × regressor
    g2 = df.groupby(['method', 'regressor']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std')).reset_index()
    pivot = g2.pivot(index='method', columns='regressor', values='r2_mean')
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False)
    print(f"\n--- Mean R² across all 6 targets (TKPI natural) ---")
    print(pivot.round(4).to_string())
    pivot.to_csv(EXP_TABLES / "downstream_tkpi_regression_summary.csv")

    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
