"""
downstream_tkpi_natural_lr.py — Downstream evaluation on TKPI (natural
missingness) using Logistic Regression instead of RandomForest.

Tests whether MissForest's downstream advantage is an artefact of
imputer–classifier coupling (both tree-based). If MissForest still leads
under a linear classifier, the finding is robust.

Output: downstream_tkpi_natural_lr.csv
        downstream_tkpi_natural_lr_summary.csv
"""
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold

from common import (load_tkpi, impute, _prep_norm,
                        EXP_TABLES, NUTRIENT_TRANSFORM_MAP)
from pica import pica_impute
from _bootstrap import PROJECT_ROOT

CLF_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']
COL_METHODS = NUTRIENT_TRANSFORM_MAP


def load_all():
    X, ids, cols = load_tkpi()
    groups = np.array([k[0] for k in ids])
    keep = groups != 'Q'
    return X[keep], ids[keep], groups[keep], cols


def impute_full(method, X_raw, cols):
    mask_train = ~np.isnan(X_raw)
    mask_test = np.zeros(X_raw.shape, dtype=bool)
    if method == 'pica':
        prep, Xin = _prep_norm(X_raw, mask_train, mask_test, col_methods=COL_METHODS)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, X_raw, mask_train, mask_test, seed=42,
                  col_methods=COL_METHODS)


def clf_scores(X, y, clf_seed):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, random_state=clf_seed,
                                  solver='lbfgs'))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=clf_seed)
    r = cross_validate(pipe, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    return float(r['test_accuracy'].mean()), float(r['test_f1_macro'].mean())


if __name__ == "__main__":
    t0 = time.time()
    X_raw, ids, y, cols = load_all()
    n_classes = len(set(y))
    n_missing = np.isnan(X_raw).sum()
    n_total = X_raw.size
    print(f"TKPI all rows: {X_raw.shape}, {n_classes} food groups, "
          f"{n_missing}/{n_total} naturally missing ({100*n_missing/n_total:.1f}%)",
          flush=True)

    # Oracle on complete-case
    comp = ~np.isnan(X_raw).any(axis=1)
    Xcc = X_raw[comp]
    y_cc = y[comp]
    oracle_acc, oracle_f1 = clf_scores(Xcc, y_cc, 0)
    print(f"Oracle (complete-case n={Xcc.shape[0]}): "
          f"acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}", flush=True)

    # Resume support
    csv_path = EXP_TABLES / "downstream_tkpi_natural_lr.csv"
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(int(r["clf_seed"]), r["method"]) for r in rows}
        print(f"Resuming: {len(done)} cells already done", flush=True)
    else:
        rows, done = [], set()

    for m in METHODS:
        ts = time.time()
        X_imp = impute_full(m, X_raw, cols)

        for clf_seed in CLF_SEEDS:
            if (clf_seed, m) in done:
                continue
            acc, f1 = clf_scores(X_imp, y, clf_seed)
            rows.append(dict(clf_seed=clf_seed, method=m,
                             clf_acc=acc, clf_f1=f1))
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        mean_acc = np.mean([r['clf_acc'] for r in rows if r['method'] == m])
        mean_f1 = np.mean([r['clf_f1'] for r in rows if r['method'] == m])
        print(f"  {m:<12} acc={mean_acc:.4f}  F1={mean_f1:.4f}  "
              f"({time.time()-ts:.0f}s)", flush=True)

    # Summary
    df = pd.DataFrame(rows)
    g = df.groupby('method').agg(
        clf_acc=('clf_acc', 'mean'), clf_acc_std=('clf_acc', 'std'),
        clf_f1=('clf_f1', 'mean'), clf_f1_std=('clf_f1', 'std'))
    g['rank_acc'] = (-g.clf_acc).rank()
    g['rank_f1'] = (-g.clf_f1).rank()
    g = g.sort_values('clf_acc', ascending=False)
    print(f"\n--- Summary (TKPI natural, LR classifier, all {X_raw.shape[0]} rows) ---")
    print(f"Oracle (complete-case n={Xcc.shape[0]}): "
          f"acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}")
    print(g.round(4).to_string())
    g.to_csv(EXP_TABLES / "downstream_tkpi_natural_lr_summary.csv")

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
