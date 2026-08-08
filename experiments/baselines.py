"""
baselines.py — Additional matrix-completion and deep-learning baselines on TKPI.

Adds, on the exact same leakage-safe pipeline as the main experiments:
  - IterativeSVD  (fancyimpute) : matrix completion beyond SoftImpute.
  - DAE           (dl)          : masked denoising autoencoder (deep learning).
  - NuclearNormMinimization     : recorded as INFEASIBLE at FCDB scale
                                  (cvxpy SDP overflows INT_MAX on 1146x19).

Runs Scenario A (20%, 10 seeds) and the missing-rate sweep (5-50%, 10 seeds) so
these methods can be placed alongside SoftImpute/KNN in the summary tables.
"""
import time
import numpy as np
import pandas as pd

from common import (
    load_tkpi, make_random_mask, impute, _prep_norm,
    MISSING_FRACS, SEEDS_10, EXP_TABLES,
)
from dl import dae_impute
from utils import evaluate_imputation

X_raw, _, cols = load_tkpi()


def run_method(method, mtr, mte, seed):
    if method == 'dae':
        prep, Xin = _prep_norm(X_raw, mtr, mte)
        Xnorm_imp = dae_impute(Xin, seed=seed)
        Ximp = prep.inverse_transform(Xnorm_imp)
    else:
        Ximp = impute(method, X_raw, mtr, mte, seed=seed)
    return evaluate_imputation(X_raw, Ximp, mte, cols)['median_nrmse']


if __name__ == "__main__":
    t0 = time.time()
    METHODS = ['iterativesvd', 'dae']

    # ── Scenario A (20%) ──────────────────────────────────────────────────
    fA = EXP_TABLES / "baselines_scenarioA_10seed.csv"
    if fA.exists() and pd.read_csv(fA).groupby('method').seed.nunique().min() >= len(SEEDS_10):
        print("Scenario A already complete, skipping.", flush=True)
    else:
        rowsA = []
        for seed in SEEDS_10:
            mtr, mte = make_random_mask(X_raw, 0.20, seed)
            for m in METHODS:
                v = run_method(m, mtr, mte, seed)
                rowsA.append(dict(seed=seed, method=m, nrmse=v))
            print(f"  [A] seed {seed} done  [{(time.time()-t0)/60:.1f}m]", flush=True)
        pd.DataFrame(rowsA).to_csv(fA, index=False)

    # ── Sweep (5-50%) ─────────────────────────────────────────────────────
    fS = EXP_TABLES / "baselines_sweep_10seed.csv"
    rows, done = [], set()
    if fS.exists():
        rows = pd.read_csv(fS).to_dict('records')
        done = {(round(r['missing_frac'], 3), int(r['seed']), r['method']) for r in rows}
        print(f"Resuming sweep: {len(done)} cells already done", flush=True)
    for frac in MISSING_FRACS:
        for seed in SEEDS_10:
            mtr, mte = make_random_mask(X_raw, frac, seed)
            for m in METHODS:
                if (round(frac, 3), seed, m) in done:
                    continue
                v = run_method(m, mtr, mte, seed)
                rows.append(dict(missing_frac=frac, seed=seed, method=m, nrmse=v))
            pd.DataFrame(rows).to_csv(fS, index=False)
        df = pd.DataFrame(rows)
        sub = df[df.missing_frac == frac]
        means = {m: sub[sub.method == m].nrmse.mean() for m in METHODS}
        print(f"[{int(frac*100):3d}%] " +
              "  ".join(f"{m}={means[m]:.4f}" for m in METHODS) +
              f"  [{(time.time()-t0)/60:.1f}m]", flush=True)

    # ── NNM infeasibility note ────────────────────────────────────────────
    (EXP_TABLES / "nnm_infeasible.txt").write_text(
        "NuclearNormMinimization (fancyimpute, exact convex SDP via cvxpy) is\n"
        "computationally infeasible on the 1146x19 TKPI matrix: it raises\n"
        "OverflowError('number of elements exceeds INT_MAX') because the SDP\n"
        "variable scales with (n*p)^2. This is the practical reason the field\n"
        "uses scalable surrogates such as SoftImpute rather than exact NNM.\n")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
