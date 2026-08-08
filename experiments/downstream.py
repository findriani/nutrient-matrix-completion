"""
downstream.py — Downstream utility & structural fidelity.

Uses the USDA complete-case subset as ground truth. For each of SEEDS_10 masks we
inject 20% MCAR, impute with each method, and evaluate three axes per method:

  1. Imputation NRMSE (held-out point error) — the paper's usual metric.
  2. Downstream food-category classification — stratified 5-fold RandomForest on the
     *imputed* matrix; report both accuracy and macro-F1 (the 25 categories are
     unbalanced). Oracle = same protocol on the true matrix.
  3. Structural fidelity — RELATIVE covariance error ||S_hat-S||_F/||S||_F and a
     RELATIVE precision error ||T_hat-T||_F/||T||_F using the same Ledoit-Wolf
     regularised estimator for the reference and every imputed matrix.

Protocol notes:
  * ONE common standardization: column mean/SD from the COMPLETE reference matrix,
    applied to the reference AND every imputed matrix, before estimating S / T.
    (Self-standardizing each matrix would hide imputation-induced variance
    distortion and turn this into a correlation-structure metric.)
  * Identical folds, RF random_state and hyperparameters for every method AND the
    oracle (CLF_SEED), so results are paired by mask; only the data (mask) varies.
  * Folds are averaged WITHIN each mask; methods are compared across the 10 masks.
  * This is a TRANSDUCTIVE completed-database use case (impute the full matrix, then
    split), not an inductive model for unseen foods.

Output: downstream_usda.csv (seed-level) + downstream_usda_summary.csv + oracle txt.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.covariance import LedoitWolf

from usda import load_usda, make_mcar_mask, USDA_TRANSFORM_MAP
from common import impute, _prep_norm, EXP_TABLES, SEEDS_10
from pica import pica_impute
from _bootstrap import PROJECT_ROOT
from utils import evaluate_imputation

HERE = PROJECT_ROOT
SEEDS = SEEDS_10
FRAC = 0.20
CLF_SEED = 0          # fixed folds + RF across every method and the oracle
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']


def load_gt():
    X, ids, cols = load_usda()
    comp = ~np.isnan(X).any(axis=1)
    Xg = X[comp]
    food = pd.read_csv(
        HERE / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food.csv",
        usecols=['fdc_id', 'food_category_id'])
    cat = (pd.Series(ids[comp]).to_frame('fdc_id')
           .merge(food, on='fdc_id', how='left').food_category_id.values)
    return Xg, cat, cols


def impute_any(method, Xg, mtr, mte, seed):
    """Dispatch: PICA needs the normalised-input path; everything else via impute()."""
    if method == 'pica':
        prep, Xin = _prep_norm(Xg, mtr, mte, col_methods=USDA_TRANSFORM_MAP)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, Xg, mtr, mte, seed=seed, col_methods=USDA_TRANSFORM_MAP)


def clf_scores(X, y):
    """Stratified 5-fold RF; identical folds/estimator for every call. Returns
    (mean accuracy, mean macro-F1) averaged over the 5 folds."""
    clf = RandomForestClassifier(n_estimators=200, random_state=CLF_SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CLF_SEED)
    r = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    return float(r['test_accuracy'].mean()), float(r['test_f1_macro'].mean())


def struct_errors(X, mu, sd, Sigma_true, nS, Theta_true, nT):
    """Relative covariance & precision Frobenius error on common-standardised data."""
    Z = (X - mu) / sd
    Sigma = np.cov(Z, rowvar=False)
    cov_relerr = float(np.linalg.norm(Sigma - Sigma_true, 'fro') / nS)
    try:
        Theta = LedoitWolf().fit(Z).precision_
        prec_relerr = float(np.linalg.norm(Theta - Theta_true, 'fro') / nT)
    except Exception:
        prec_relerr = np.nan
    return cov_relerr, prec_relerr


if __name__ == "__main__":
    t0 = time.time()
    Xg, y, cols = load_gt()
    print(f"Ground-truth complete-case: {Xg.shape}, {len(set(y))} categories", flush=True)

    # common reference standardization + true structure (Ledoit-Wolf precision)
    mu, sd = Xg.mean(0), Xg.std(0) + 1e-12
    Zt = (Xg - mu) / sd
    Sigma_true = np.cov(Zt, rowvar=False)
    nS = np.linalg.norm(Sigma_true, 'fro')
    Theta_true = LedoitWolf().fit(Zt).precision_
    nT = np.linalg.norm(Theta_true, 'fro')

    oracle_acc, oracle_f1 = clf_scores(Xg, y)
    print(f"Oracle (true matrix): acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}", flush=True)

    # resume-safe: skip (seed, method) cells already checkpointed to the CSV
    _dpath = EXP_TABLES / "downstream_usda.csv"
    if _dpath.exists():
        rows = pd.read_csv(_dpath).to_dict("records")
        done = {(int(r["seed"]), r["method"]) for r in rows}
        print(f"Resuming: {len(done)} (seed,method) cells already done", flush=True)
    else:
        rows, done = [], set()
    for seed in SEEDS:
        mtr, mte = make_mcar_mask(Xg, FRAC, seed)
        for m in METHODS:
            if (seed, m) in done:
                continue
            ts = time.time()
            X_imp = impute_any(m, Xg, mtr, mte, seed)
            nrmse = evaluate_imputation(Xg, X_imp, mte, cols)['median_nrmse']
            acc, f1 = clf_scores(X_imp, y)
            cov_re, prec_re = struct_errors(X_imp, mu, sd, Sigma_true, nS, Theta_true, nT)
            rows.append(dict(seed=seed, method=m, nrmse=nrmse, clf_acc=acc, clf_f1=f1,
                             cov_relerr=cov_re, prec_relerr=prec_re))
            print(f"  seed{seed} {m:<12} NRMSE={nrmse:.4f} acc={acc:.4f} F1={f1:.4f} "
                  f"covRE={cov_re:.4f} precRE={prec_re:.4f} ({time.time()-ts:.0f}s)",
                  flush=True)
        pd.DataFrame(rows).to_csv(EXP_TABLES / "downstream_usda.csv", index=False)
        print(f"  -- seed {seed} done  [{(time.time()-t0)/60:.1f}m]", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby('method').agg(
        nrmse=('nrmse', 'mean'), clf_acc=('clf_acc', 'mean'),
        clf_f1=('clf_f1', 'mean'), cov_relerr=('cov_relerr', 'mean'),
        prec_relerr=('prec_relerr', 'mean'))
    g['rank_nrmse'] = g.nrmse.rank()
    g['rank_acc'] = (-g.clf_acc).rank()
    g['rank_f1'] = (-g.clf_f1).rank()
    g['rank_cov'] = g.cov_relerr.rank()
    g['rank_prec'] = g.prec_relerr.rank()
    g = g.sort_values('nrmse')
    print(f"\nOracle acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}")
    print(g.round(4).to_string())
    g.to_csv(EXP_TABLES / "downstream_usda_summary.csv")
    Path(EXP_TABLES / "downstream_oracle.txt").write_text(
        f"oracle_clf_acc={oracle_acc:.4f}\noracle_clf_macroF1={oracle_f1:.4f}\n")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
