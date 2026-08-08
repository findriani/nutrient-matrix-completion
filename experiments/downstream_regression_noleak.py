"""
downstream_regression_noleak.py — Leakage-free leave-one-nutrient-out
regression on TKPI and USDA with natural missingness.

The target nutrient column is REMOVED from the
matrix BEFORE imputation, so no imputer can use target values to inform
predictor imputation. This eliminates the information pathway where
multivariate imputers (MICE, SoftImpute, MissForest, PICA) leak target
information into imputed predictor columns.

Protocol per (dataset, method, target):
  1. Remove the target column from X_raw  (n × p  →  n × (p-1))
  2. Adjust col_methods to match the reduced column count
  3. Impute the (p-1)-column matrix
  4. Select rows where the target is originally observed
  5. Train regressors on imputed features → observed target
  6. Evaluate via 5-fold CV with 5 seeds

Targets: Protein, Fat, Carbohydrate, Calcium, Iron, Vitamin_C
Regressors: RF (200 trees), Ridge (alpha=1), KNR (k=5)
"""
import sys, os, time
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
from usda import load_usda, USDA_TRANSFORM_MAP
from pica import pica_impute
from _bootstrap import PROJECT_ROOT

CV_SEEDS = [0, 1, 2, 3, 4]
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']

TARGETS = {
    'Protein':      'macro',
    'Fat':          'macro',
    'Carbohydrate': 'macro',
    'Calcium':      'micro',
    'Iron':         'micro',
    'Vitamin_C':    'micro',
}

REGRESSORS = ['rf', 'ridge', 'knr']


def impute_without_target(method, X_raw, col_methods, seed=42):
    """Impute a matrix that already has the target column removed."""
    mask_train = ~np.isnan(X_raw)
    mask_test = np.zeros(X_raw.shape, dtype=bool)
    if method == 'pica':
        prep, Xin = _prep_norm(X_raw, mask_train, mask_test,
                               col_methods=col_methods)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, X_raw, mask_train, mask_test, seed=seed,
                  col_methods=col_methods)


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


def run_dataset(dataset_name, X_raw, cols, col_methods_full):
    """Run leakage-free regression for one dataset."""
    col_idx = {c: i for i, c in enumerate(cols)}
    print(f"\n{'='*60}")
    print(f"  {dataset_name}: {X_raw.shape}")
    print(f"{'='*60}")

    for tgt in TARGETS:
        n_obs = int(np.sum(~np.isnan(X_raw[:, col_idx[tgt]])))
        print(f"  {tgt:15s}: {n_obs} observed rows", flush=True)

    # Resume support
    csv_path = EXP_TABLES / f"downstream_{dataset_name.lower()}_regression_noleak.csv"
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["method"], r["target"], r["regressor"], int(r["cv_seed"]))
                for r in rows}
        print(f"Resuming: {len(done)} cells already done", flush=True)
    else:
        rows, done = [], set()

    t0 = time.time()

    for m in METHODS:
        print(f"\n  Method: {m}", flush=True)

        for tgt, tgt_type in TARGETS.items():
            ti = col_idx[tgt]

            # Check if all cells for this (method, target) are already done
            all_done = all(
                (m, tgt, rt, cs) in done
                for rt in REGRESSORS for cs in CV_SEEDS
            )
            if all_done:
                print(f"    {tgt:15s}: already done", flush=True)
                continue

            # === LEAKAGE FIX ===
            # Remove target column BEFORE imputation
            feat_col_idx = [i for i in range(len(cols)) if i != ti]
            X_reduced = X_raw[:, feat_col_idx]
            cm_reduced = [col_methods_full[i] for i in feat_col_idx]

            ts = time.time()
            X_imp = impute_without_target(m, X_reduced, cm_reduced)
            imp_time = time.time() - ts

            # Rows where target is observed (ground truth)
            obs_mask = ~np.isnan(X_raw[:, ti])
            y = X_raw[obs_mask, ti]
            X_feat = X_imp[obs_mask]  # all columns are features (target removed)

            for rt in REGRESSORS:
                for cv_seed in CV_SEEDS:
                    if (m, tgt, rt, cv_seed) in done:
                        continue
                    r2, rmse = reg_scores(X_feat, y, rt, cv_seed)
                    rows.append(dict(method=m, target=tgt, target_type=tgt_type,
                                     regressor=rt, cv_seed=cv_seed,
                                     r2=r2, rmse=rmse))
                    done.add((m, tgt, rt, cv_seed))

                r2s = [r['r2'] for r in rows
                       if r['method'] == m and r['target'] == tgt
                       and r['regressor'] == rt]
                print(f"    {tgt:15s} {rt:5s}: R2={np.mean(r2s):.4f}  "
                      f"(imp {imp_time:.1f}s)", flush=True)

            # Save after each target (resume-safe)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        print(f"    [{(time.time()-t0)/60:.1f}m elapsed]", flush=True)

    # --- Summaries ---
    df = pd.DataFrame(rows)

    # Detail: per method × target × regressor
    g = df.groupby(['method', 'target', 'regressor']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        rmse_mean=('rmse', 'mean')).reset_index()
    g.to_csv(EXP_TABLES / f"downstream_{dataset_name.lower()}_regression_noleak_detail.csv",
             index=False)

    # Compact: mean R2 across all 6 targets per method × regressor
    g2 = df.groupby(['method', 'regressor']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std')).reset_index()
    pivot = g2.pivot(index='method', columns='regressor', values='r2_mean')
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False)
    print(f"\n--- {dataset_name} Leakage-Free Regression Summary ---")
    print(pivot.round(4).to_string())
    pivot.to_csv(EXP_TABLES / f"downstream_{dataset_name.lower()}_regression_noleak_summary.csv")

    # Also print per-target detail for manuscript
    print(f"\n--- {dataset_name} Per-Target R2 (mean across seeds) ---")
    detail_pivot = g.pivot_table(index='method', columns=['target', 'regressor'],
                                  values='r2_mean')
    print(detail_pivot.round(3).to_string())

    return pivot


if __name__ == "__main__":
    t_start = time.time()

    # ── TKPI ──
    X_tkpi, ids_tkpi, cols_tkpi = load_tkpi()
    groups = np.array([k[0] for k in ids_tkpi])
    keep = groups != 'Q'
    X_tkpi = X_tkpi[keep]

    tkpi_summary = run_dataset('TKPI', X_tkpi, cols_tkpi,
                               NUTRIENT_TRANSFORM_MAP)

    # ── USDA ──
    X_usda, ids_usda, cols_usda = load_usda()
    usda_summary = run_dataset('USDA', X_usda, cols_usda,
                               USDA_TRANSFORM_MAP)

    # ── Comparison with original (leaky) results ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Original vs Leakage-Free")
    print(f"{'='*60}")

    for ds in ['tkpi', 'usda']:
        orig_path = EXP_TABLES / f"downstream_{ds}_regression_summary.csv"
        new_path = EXP_TABLES / f"downstream_{ds}_regression_noleak_summary.csv"
        if orig_path.exists() and new_path.exists():
            orig = pd.read_csv(orig_path, index_col=0)
            new = pd.read_csv(new_path, index_col=0)
            # Align columns
            shared_cols = sorted(set(orig.columns) & set(new.columns))
            print(f"\n  {ds.upper()} — avg R2 change (noleak - original):")
            for m in orig.index:
                if m in new.index and 'avg' in shared_cols:
                    delta = new.loc[m, 'avg'] - orig.loc[m, 'avg']
                    print(f"    {m:15s}: {delta:+.4f}  "
                          f"(orig={orig.loc[m, 'avg']:.4f}, "
                          f"noleak={new.loc[m, 'avg']:.4f})")

    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min", flush=True)
