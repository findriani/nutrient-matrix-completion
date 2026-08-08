"""
repin.py — regenerate the SoftImpute / IterativeSVD rows of already-computed
tables under the deterministic (seed-pinned) pipeline, so every reported number is
exactly reproducible.

Only SoftImpute and IterativeSVD were nondeterministic (unseeded randomized SVD /
TruncatedSVD); compat_patch now pins both. Every other method already fixes its seed,
so we recompute ONLY these two methods' rows — with each table's exact masking — and
splice them back, leaving the slow, already-deterministic tree methods untouched.

Prints the max |NRMSE delta| per table (expected small; documents the change).
Tables handled: sweep_tkpi, sweep_usda, main_scenarioA/B, per_nutrient_A,
baselines_scenarioA, baselines_sweep. (Derived stats — wilcoxon_per_rate — are
regenerated separately after this.)
"""
import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401  puts project root on sys.path
from common import (load_tkpi, make_random_mask, impute, impute_and_nrmse,
                        EXP_TABLES)
from utils import evaluate_imputation
from usda import load_usda, make_mcar_mask, USDA_TRANSFORM_MAP
from main_tkpi import block_micro_mask   # reuses micro_idx; import runs no __main__

T = EXP_TABLES


def repin_sweep(fname, load_fn, mask_fn, method, col_methods=None):
    """Uniform schema: missing_frac, seed, method, nrmse."""
    X, _, cols = load_fn()
    path = T / fname
    df = pd.read_csv(path)
    d = []
    for i, r in df[df.method == method].iterrows():
        mtr, mte = mask_fn(X, float(r.missing_frac), int(r.seed))
        new = impute_and_nrmse(method, X, mtr, mte, cols, seed=int(r.seed),
                               col_methods=col_methods)
        d.append(abs(new - r.nrmse)); df.at[i, 'nrmse'] = new
    df.to_csv(path, index=False)
    return max(d) if d else 0.0, len(d)


def repin_scenarioA_and_per_nutrient():
    """scenarioA schema: seed, method, median_nrmse, mean_mae; plus wide per-nutrient."""
    X, _, cols = load_tkpi()
    dfa = pd.read_csv(T / "main_scenarioA_10seed.csv")
    pn = pd.read_csv(T / "per_nutrient_A_10seed.csv")
    per_seed_pernut = {c: [] for c in cols}
    d = []
    for i, r in dfa[dfa.method == 'softimpute'].iterrows():
        mtr, mte = make_random_mask(X, 0.20, int(r.seed))
        X_imp = impute('softimpute', X, mtr, mte, seed=int(r.seed))
        res = evaluate_imputation(X, X_imp, mte, cols)['per_nutrient']
        nl = [res[c]['nrmse'] for c in cols if c in res]
        ml = [res[c]['mae'] for c in cols if c in res]
        d.append(abs(float(np.median(nl)) - r.median_nrmse))
        dfa.at[i, 'median_nrmse'] = float(np.median(nl))
        dfa.at[i, 'mean_mae'] = float(np.mean(ml))
        for c in cols:
            if c in res:
                per_seed_pernut[c].append(res[c]['nrmse'])
    dfa.to_csv(T / "main_scenarioA_10seed.csv", index=False)
    # update the softimpute column of the wide per-nutrient table (mean over seeds)
    pn['softimpute'] = [float(np.mean(per_seed_pernut[c])) if per_seed_pernut[c]
                        else np.nan for c in pn['nutrient']]
    pn.to_csv(T / "per_nutrient_A_10seed.csv", index=False)
    return max(d) if d else 0.0, len(d)


def repin_scenarioB():
    X, _, cols = load_tkpi()
    dfb = pd.read_csv(T / "main_scenarioB_10seed.csv")
    d = []
    for i, r in dfb[dfb.method == 'softimpute'].iterrows():
        mtr, mte = block_micro_mask(X, 0.30, int(r.seed))
        X_imp = impute('softimpute', X, mtr, mte, seed=int(r.seed))
        res = evaluate_imputation(X, X_imp, mte, cols)['per_nutrient']
        nl = [res[c]['nrmse'] for c in cols if c in res]
        ml = [res[c]['mae'] for c in cols if c in res]
        d.append(abs(float(np.median(nl)) - r.median_nrmse))
        dfb.at[i, 'median_nrmse'] = float(np.median(nl))
        dfb.at[i, 'mean_mae'] = float(np.mean(ml))
    dfb.to_csv(T / "main_scenarioB_10seed.csv", index=False)
    return max(d) if d else 0.0, len(d)


def repin_baselines_scenarioA():
    """schema: seed, method, nrmse  (iterativesvd + dae)."""
    X, _, cols = load_tkpi()
    path = T / "baselines_scenarioA_10seed.csv"
    df = pd.read_csv(path)
    d = []
    for i, r in df[df.method == 'iterativesvd'].iterrows():
        mtr, mte = make_random_mask(X, 0.20, int(r.seed))
        new = impute_and_nrmse('iterativesvd', X, mtr, mte, cols, seed=int(r.seed))
        d.append(abs(new - r.nrmse)); df.at[i, 'nrmse'] = new
    df.to_csv(path, index=False)
    return max(d) if d else 0.0, len(d)


if __name__ == "__main__":
    print("Repinning SoftImpute / IterativeSVD rows under deterministic SVD...\n")
    results = []
    results.append(("sweep_tkpi  softimpute",
                    *repin_sweep("sweep_tkpi_10seed.csv", load_tkpi, make_random_mask,
                                 'softimpute')))
    results.append(("sweep_usda  softimpute",
                    *repin_sweep("sweep_usda_10seed.csv", load_usda, make_mcar_mask,
                                 'softimpute', col_methods=USDA_TRANSFORM_MAP)))
    results.append(("baselines_sweep isvd",
                    *repin_sweep("baselines_sweep_10seed.csv", load_tkpi,
                                 make_random_mask, 'iterativesvd')))
    results.append(("scenarioA+pernut softimpute", *repin_scenarioA_and_per_nutrient()))
    results.append(("scenarioB  softimpute", *repin_scenarioB()))
    results.append(("baselines_scenarioA isvd", *repin_baselines_scenarioA()))

    print(f"\n{'table':<32}{'rows':>6}{'max|delta|':>14}")
    for name, mx, n in results:
        print(f"{name:<32}{n:>6}{mx:>14.5f}")
    print("\nDONE. All SoftImpute/IterativeSVD rows now come from the pinned pipeline.")
