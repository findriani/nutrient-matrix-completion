"""
nmf_rank_diag.py — Masked-NMF rank-sensitivity diagnostic.

Verifies that the downstream collapse (acc ~0.38, precRE ~927) is not a rank-2
artefact. Runs masked NMF at ranks 2, 3, 4, 5 on one USDA seed (seed=123, 20%
MCAR) and reports: NRMSE, classification accuracy, macro-F1, relative covariance
error, relative precision error, and the condition number of the imputed
covariance matrix.

If the collapse persists across ranks, it is a property of the masked-NMF
formulation, not the rank choice. If higher ranks fix it, the manuscript should
say "the tested rank-2 configuration" rather than "masked NMF."
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.covariance import LedoitWolf

from _bootstrap import PROJECT_ROOT  # noqa: F401
from usda import load_usda, make_mcar_mask, USDA_TRANSFORM_MAP
from common import _prep_norm, masked_nmf, EXP_TABLES
from utils import evaluate_imputation

SEED = 123
FRAC = 0.20
CLF_SEED = 0
RANKS = [2, 3, 4, 5]


def load_gt():
    import pandas as pd
    X, ids, cols = load_usda()
    comp = ~np.isnan(X).any(axis=1)
    Xg = X[comp]
    food = pd.read_csv(
        PROJECT_ROOT / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food.csv",
        usecols=['fdc_id', 'food_category_id'])
    cat = (pd.Series(ids[comp]).to_frame('fdc_id')
           .merge(food, on='fdc_id', how='left').food_category_id.values)
    return Xg, cat, cols


def impute_nmf(Xg, mtr, mte, rank):
    """Run masked NMF at a given rank and return the imputed matrix."""
    prep, X_input = _prep_norm(Xg, mtr, mte, col_methods=USDA_TRANSFORM_MAP)
    mask_obs = ~np.isnan(X_input)
    X_for_nmf = np.maximum(np.where(mask_obs, X_input, 0.0), 0.0)
    X_imp_norm = masked_nmf(X_for_nmf, mask_obs, rank=rank, seed=SEED)
    return prep.inverse_transform(X_imp_norm)


if __name__ == "__main__":
    t0 = time.time()
    Xg, y, cols = load_gt()
    mtr, mte = make_mcar_mask(Xg, FRAC, SEED)

    # Reference standardization + true structure
    mu, sd = Xg.mean(0), Xg.std(0) + 1e-12
    Zt = (Xg - mu) / sd
    Sigma_true = np.cov(Zt, rowvar=False)
    nS = np.linalg.norm(Sigma_true, 'fro')
    Theta_true = LedoitWolf().fit(Zt).precision_
    nT = np.linalg.norm(Theta_true, 'fro')

    clf = RandomForestClassifier(n_estimators=200, random_state=CLF_SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CLF_SEED)

    print(f"USDA complete-case {Xg.shape}, seed={SEED}, frac={FRAC}")
    print(f"{'rank':>5}  {'NRMSE':>7}  {'acc':>7}  {'F1':>7}  "
          f"{'covRE':>7}  {'precRE':>9}  {'cond(Sig)':>10}")
    print("-" * 65)

    for rank in RANKS:
        ts = time.time()
        X_imp = impute_nmf(Xg, mtr, mte, rank)
        nrmse = evaluate_imputation(Xg, X_imp, mte, cols)['median_nrmse']

        r = cross_validate(clf, X_imp, y, cv=cv, scoring=['accuracy', 'f1_macro'])
        acc = float(r['test_accuracy'].mean())
        f1 = float(r['test_f1_macro'].mean())

        Z = (X_imp - mu) / sd
        Sigma = np.cov(Z, rowvar=False)
        cov_re = float(np.linalg.norm(Sigma - Sigma_true, 'fro') / nS)
        cond = float(np.linalg.cond(Sigma))
        try:
            Theta = LedoitWolf().fit(Z).precision_
            prec_re = float(np.linalg.norm(Theta - Theta_true, 'fro') / nT)
        except Exception:
            prec_re = float('nan')

        print(f"{rank:>5}  {nrmse:>7.4f}  {acc:>7.4f}  {f1:>7.4f}  "
              f"{cov_re:>7.4f}  {prec_re:>9.1f}  {cond:>10.1f}  "
              f"({time.time()-ts:.0f}s)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
