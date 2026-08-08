"""
downstream_tkpi.py — Downstream evaluation on TKPI at 20%, 30%, 40% MCAR.

Same protocol as downstream.py but on TKPI (1146×19). Food groups are
extracted from the KODE prefix (first letter → 13 groups; singleton Q dropped
→ 12 classes, 1145 rows).

Output per rate: downstream_tkpi_{pct}pct.csv (seed-level)
                 downstream_tkpi_{pct}pct_summary.csv
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.covariance import LedoitWolf

from common import (load_tkpi, make_random_mask, impute, _prep_norm,
                        EXP_TABLES, SEEDS_10, NUTRIENT_TRANSFORM_MAP)
from pica import pica_impute
from _bootstrap import PROJECT_ROOT
from utils import evaluate_imputation

SEEDS = SEEDS_10
FRACS = [0.20, 0.30, 0.40]
CLF_SEED = 0
METHODS = ['mean', 'median', 'knn_k10', 'softimpute', 'masked_nmf',
           'mice', 'iterativesvd', 'missforest', 'pica']
COL_METHODS = NUTRIENT_TRANSFORM_MAP


def load_gt():
    """Load TKPI complete-case rows with first-letter food-group labels.
    Drops the singleton Q group (1 item can't stratify)."""
    X, ids, cols = load_tkpi()
    comp = ~np.isnan(X).any(axis=1)
    Xg = X[comp]
    ids_comp = ids[comp]
    # first letter of KODE = food group
    groups = np.array([k[0] for k in ids_comp])
    # drop singleton Q
    keep = groups != 'Q'
    return Xg[keep], groups[keep], cols


def impute_any(method, Xg, mtr, mte, seed):
    if method == 'pica':
        prep, Xin = _prep_norm(Xg, mtr, mte, col_methods=COL_METHODS)
        return prep.inverse_transform(pica_impute(Xin))
    return impute(method, Xg, mtr, mte, seed=seed, col_methods=COL_METHODS)


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
    n_classes = len(set(y))
    print(f"TKPI complete-case: {Xg.shape}, {n_classes} food groups", flush=True)
    print(f"Group distribution: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

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
        print(f"  FRAC = {pct}% MCAR  (TKPI)")
        print(f"{'='*60}", flush=True)

        csv_path = EXP_TABLES / f"downstream_tkpi_{pct}pct.csv"
        if csv_path.exists():
            rows = pd.read_csv(csv_path).to_dict("records")
            done = {(int(r["seed"]), r["method"]) for r in rows}
            print(f"Resuming: {len(done)} cells already done", flush=True)
        else:
            rows, done = [], set()

        for seed in SEEDS:
            mtr, mte = make_random_mask(Xg, frac, seed)
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
                # checkpoint after every method
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
        print(f"\n--- Summary at {pct}% (TKPI) ---")
        print(f"Oracle acc={oracle_acc:.4f}  macroF1={oracle_f1:.4f}")
        print(g.round(4).to_string())
        g.to_csv(EXP_TABLES / f"downstream_tkpi_{pct}pct_summary.csv")

    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
