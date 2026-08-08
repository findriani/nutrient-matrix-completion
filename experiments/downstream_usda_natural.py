"""
downstream_usda_natural.py — Downstream evaluation on USDA with natural
missingness, using three classifiers (RF, LR, NB).

Uses ALL 7793 USDA rows with their real missing-value pattern. Each method
imputes the full matrix once; downstream quality is measured via food-category
classification (25 categories from food.csv) with three classifier families
to test for imputer–classifier coupling.

Classifiers:
  RF  — RandomForest (200 trees, n_jobs=-1)
  LR  — Logistic Regression (StandardScaler + lbfgs, max_iter=2000)
  NB  — Gaussian Naive Bayes (StandardScaler)

Structural fidelity: covariance/precision error relative to complete-case
reference (n=5114).

Output per classifier: downstream_usda_natural_{clf}.csv (seed-level)
                       downstream_usda_natural_{clf}_summary.csv
         Also: downstream_usda_natural_struct.csv (structural metrics, one per method)
"""
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.covariance import LedoitWolf

from usda import load_usda, USDA_TRANSFORM_MAP
from common import impute, _prep_norm, EXP_TABLES
from pica import pica_impute
from _bootstrap import PROJECT_ROOT

CLF_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']
COL_METHODS = USDA_TRANSFORM_MAP
CLASSIFIERS = ['rf', 'lr', 'nb']


def load_all():
    """Load all USDA rows with food-category labels."""
    X, ids, cols = load_usda()
    food = pd.read_csv(
        PROJECT_ROOT / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food.csv",
        usecols=['fdc_id', 'food_category_id'])
    cat = (pd.Series(ids).to_frame('fdc_id')
           .merge(food, on='fdc_id', how='left').food_category_id.values)
    # drop rows with missing category
    valid = ~pd.isna(cat)
    return X[valid], cat[valid].astype(int), cols


def impute_full(method, X_raw):
    mask_train = ~np.isnan(X_raw)
    mask_test = np.zeros(X_raw.shape, dtype=bool)
    if method == 'pica':
        prep, Xin = _prep_norm(X_raw, mask_train, mask_test, col_methods=COL_METHODS)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, X_raw, mask_train, mask_test, seed=42,
                  col_methods=COL_METHODS)


def make_clf(clf_type, clf_seed):
    if clf_type == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=clf_seed,
                                      n_jobs=-1)
    elif clf_type == 'lr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, random_state=clf_seed,
                                      solver='lbfgs'))
        ])
    elif clf_type == 'nb':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('nb', GaussianNB())
        ])


def clf_scores(X, y, clf_type, clf_seed):
    clf = make_clf(clf_type, clf_seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=clf_seed)
    r = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    return float(r['test_accuracy'].mean()), float(r['test_f1_macro'].mean())


def struct_errors(X, mu, sd, Sigma_ref, nS, Theta_ref, nT):
    Z = (X - mu) / sd
    Sigma = np.cov(Z, rowvar=False)
    cov_re = float(np.linalg.norm(Sigma - Sigma_ref, 'fro') / nS)
    try:
        Theta = LedoitWolf().fit(Z).precision_
        prec_re = float(np.linalg.norm(Theta - Theta_ref, 'fro') / nT)
    except Exception:
        prec_re = float('nan')
    return cov_re, prec_re


if __name__ == "__main__":
    t0 = time.time()
    X_raw, y, cols = load_all()
    n_classes = len(set(y))
    n_missing = np.isnan(X_raw).sum()
    n_total = X_raw.size
    print(f"USDA all rows: {X_raw.shape}, {n_classes} categories, "
          f"{n_missing}/{n_total} naturally missing ({100*n_missing/n_total:.1f}%)",
          flush=True)

    # Reference from complete-case subset
    comp = ~np.isnan(X_raw).any(axis=1)
    Xcc = X_raw[comp]
    y_cc = y[comp]
    mu_cc, sd_cc = Xcc.mean(0), Xcc.std(0) + 1e-12
    Zt = (Xcc - mu_cc) / sd_cc
    Sigma_ref = np.cov(Zt, rowvar=False)
    nS = np.linalg.norm(Sigma_ref, 'fro')
    Theta_ref = LedoitWolf().fit(Zt).precision_
    nT = np.linalg.norm(Theta_ref, 'fro')
    print(f"Complete-case reference: {Xcc.shape[0]} rows", flush=True)

    # Oracle for each classifier
    for ct in CLASSIFIERS:
        oa, of1 = clf_scores(Xcc, y_cc, ct, 0)
        print(f"Oracle {ct.upper()} (complete-case): acc={oa:.4f}  F1={of1:.4f}",
              flush=True)

    # Per-classifier resume state
    clf_state = {}
    for ct in CLASSIFIERS:
        csv_path = EXP_TABLES / f"downstream_usda_natural_{ct}.csv"
        if csv_path.exists():
            rows = pd.read_csv(csv_path).to_dict("records")
            done = {(int(r["clf_seed"]), r["method"]) for r in rows}
            print(f"  {ct.upper()}: resuming, {len(done)} cells done", flush=True)
        else:
            rows, done = [], set()
        clf_state[ct] = {'rows': rows, 'done': done, 'path': csv_path}

    struct_rows = []

    for m in METHODS:
        ts = time.time()
        print(f"\n  Imputing {m}...", end="", flush=True)
        X_imp = impute_full(m, X_raw)
        print(f" done ({time.time()-ts:.0f}s)", flush=True)

        # Structural fidelity
        cov_re, prec_re = struct_errors(X_imp, mu_cc, sd_cc,
                                        Sigma_ref, nS, Theta_ref, nT)
        struct_rows.append(dict(method=m, cov_relerr=cov_re, prec_relerr=prec_re))

        # Classification with each classifier
        for ct in CLASSIFIERS:
            state = clf_state[ct]
            for clf_seed in CLF_SEEDS:
                if (clf_seed, m) in state['done']:
                    continue
                acc, f1 = clf_scores(X_imp, y, ct, clf_seed)
                state['rows'].append(dict(clf_seed=clf_seed, method=m,
                                          clf_acc=acc, clf_f1=f1))
                pd.DataFrame(state['rows']).to_csv(state['path'], index=False)

            accs = [r['clf_acc'] for r in state['rows'] if r['method'] == m]
            f1s = [r['clf_f1'] for r in state['rows'] if r['method'] == m]
            print(f"    {ct.upper()}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"F1={np.mean(f1s):.4f}", flush=True)

        print(f"    struct: covRE={cov_re:.4f}  precRE={prec_re:.4f}  "
              f"[{(time.time()-t0)/60:.1f}m total]", flush=True)

    # Save structural metrics
    pd.DataFrame(struct_rows).to_csv(
        EXP_TABLES / "downstream_usda_natural_struct.csv", index=False)

    # Summaries per classifier
    for ct in CLASSIFIERS:
        df = pd.DataFrame(clf_state[ct]['rows'])
        g = df.groupby('method').agg(
            clf_acc=('clf_acc', 'mean'), clf_acc_std=('clf_acc', 'std'),
            clf_f1=('clf_f1', 'mean'), clf_f1_std=('clf_f1', 'std'))
        g['rank_acc'] = (-g.clf_acc).rank()
        g['rank_f1'] = (-g.clf_f1).rank()
        g = g.sort_values('clf_acc', ascending=False)
        print(f"\n--- {ct.upper()} Summary (USDA natural, all {X_raw.shape[0]} rows) ---")
        print(g.round(4).to_string())
        g.to_csv(EXP_TABLES / f"downstream_usda_natural_{ct}_summary.csv")

    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
