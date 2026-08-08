"""
downstream_rates.py — Downstream evaluation at 30% and 40% MCAR.

Same protocol as downstream.py (20%), extended to higher missing rates
where SoftImpute's point-error advantage is established. Tests whether that
advantage carries through to classification and structural fidelity.

Output per rate: downstream_usda_{pct}pct.csv (seed-level)
                 downstream_usda_{pct}pct_summary.csv
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

SEEDS = SEEDS_10
FRACS = [0.30, 0.40]
CLF_SEED = 0
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']


def load_gt():
    X, ids, cols = load_usda()
    comp = ~np.isnan(X).any(axis=1)
    Xg = X[comp]
    food = pd.read_csv(
        PROJECT_ROOT / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food.csv",
        usecols=['fdc_id', 'food_category_id'])
    cat = (pd.Series(ids[comp]).to_frame('fdc_id')
           .merge(food, on='fdc_id', how='left').food_category_id.values)
    return Xg, cat, cols


def impute_any(method, Xg, mtr, mte, seed):
    if method == 'pica':
        prep, Xin = _prep_norm(Xg, mtr, mte, col_methods=USDA_TRANSFORM_MAP)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, Xg, mtr, mte, seed=seed, col_methods=USDA_TRANSFORM_MAP)


def clf_scores(X, y):
    clf = RandomForestClassifier(n_estimators=200, random_state=CLF_SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CLF_SEED)
    r = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    return float(r['test_accuracy'].mean()), float(r['test_f1_macro'].mean())


def struct_errors(X, mu, sd, Sigma_true, nS, Theta_true, nT):
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

    mu, sd = Xg.mean(0), Xg.std(0) + 1e-12
    Zt = (Xg - mu) / sd
    Sigma_true = np.cov(Zt, rowvar=False)
    nS = np.linalg.norm(Sigma_true, 'fro')
    Theta_true = LedoitWolf().fit(Zt).precision_
    nT = np.linalg.norm(Theta_true, 'fro')

    oracle_acc, oracle_f1 = clf_scores(Xg, y)
    print(f"Oracle (true matrix): acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}", flush=True)

    for frac in FRACS:
        pct = int(frac * 100)
        print(f"\n{'='*60}")
        print(f"  FRAC = {pct}% MCAR")
        print(f"{'='*60}", flush=True)

        csv_path = EXP_TABLES / f"downstream_usda_{pct}pct.csv"
        if csv_path.exists():
            rows = pd.read_csv(csv_path).to_dict("records")
            done = {(int(r["seed"]), r["method"]) for r in rows}
            print(f"Resuming: {len(done)} cells already done", flush=True)
        else:
            rows, done = [], set()

        for seed in SEEDS:
            mtr, mte = make_mcar_mask(Xg, frac, seed)
            for m in METHODS:
                if (seed, m) in done:
                    continue
                ts = time.time()
                X_imp = impute_any(m, Xg, mtr, mte, seed)
                nrmse = evaluate_imputation(Xg, X_imp, mte, cols)['median_nrmse']
                acc, f1 = clf_scores(X_imp, y)
                cov_re, prec_re = struct_errors(X_imp, mu, sd,
                                                Sigma_true, nS, Theta_true, nT)
                rows.append(dict(seed=seed, method=m, frac=frac,
                                 nrmse=nrmse, clf_acc=acc, clf_f1=f1,
                                 cov_relerr=cov_re, prec_relerr=prec_re))
                # checkpoint after EVERY method so at most one cell is lost on crash
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  seed{seed} {m:<12} NRMSE={nrmse:.4f} acc={acc:.4f} "
                      f"F1={f1:.4f} covRE={cov_re:.4f} precRE={prec_re:.4f} "
                      f"({time.time()-ts:.0f}s)", flush=True)
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
        print(f"\n--- Summary at {pct}% ---")
        print(f"Oracle acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}")
        print(g.round(4).to_string())
        g.to_csv(EXP_TABLES / f"downstream_usda_{pct}pct_summary.csv")

    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
