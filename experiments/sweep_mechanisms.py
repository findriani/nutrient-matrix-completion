"""
sweep_mechanisms.py — crossover under MCAR / MAR / MNAR on USDA SR Legacy.

Repeats the KNN-vs-SoftImpute missing-rate sweep under each missingness mechanism to
test whether the ~25% crossover is mechanism-sensitive.

Two normalizations are reported per (mechanism, rate, seed, method):
  * nrmse          — held-out normalization: RMSE / SD(held-out truth) per nutrient,
                     median across nutrients. The paper's usual metric.
  * nrmse_refnorm  — reference normalization: RMSE / SD(full observed column) per
                     nutrient, median across nutrients. Under low-value (MNAR)
                     censoring the held-out truth spans only the low tail, so its SD
                     is small and inflates the held-out NRMSE; the reference SD is the
                     honest denominator and separates a real imputation failure from a
                     denominator artefact.

Output: sweep_usda_mechanisms_10seed.csv
        (mechanism, missing_frac, seed, method, nrmse, nrmse_refnorm)
"""
import time
import numpy as np
import pandas as pd

from usda import (load_usda, make_mcar_mask, make_mar_mask, make_mnar_mask,
                      USDA_TRANSFORM_MAP)
from common import impute, MISSING_FRACS, SEEDS_10, EXP_TABLES

OUT = EXP_TABLES / "sweep_usda_mechanisms_10seed.csv"
METHODS = ['knn_k10', 'knn_k5', 'softimpute']
MECHS = {'mcar': make_mcar_mask, 'mar': make_mar_mask, 'mnar': make_mnar_mask}


def both_nrmse(X_raw, X_imp, mask_test, ncol):
    """Median-over-nutrients NRMSE under held-out and full-column-reference norms."""
    heldout, refnorm = [], []
    for j in range(ncol):
        idx = mask_test[:, j]
        if idx.sum() == 0:
            continue
        yt, yp = X_raw[idx, j], X_imp[idx, j]
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        heldout.append(rmse / (np.std(yt) + 1e-10))
        col = X_raw[:, j]
        refnorm.append(rmse / (np.std(col[~np.isnan(col)]) + 1e-10))
    return float(np.median(heldout)), float(np.median(refnorm))


if __name__ == "__main__":
    X_raw, _, cols = load_usda()
    ncol = len(cols)
    print(f"USDA {X_raw.shape}  mechanisms={list(MECHS)}  seeds={SEEDS_10}", flush=True)

    # resume-safe: skip (mechanism, frac, seed, method) rows already checkpointed
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
        done = {(r["mechanism"], round(float(r["missing_frac"]), 4), int(r["seed"]),
                 r["method"]) for r in rows}
        print(f"Resuming: {len(done)} rows already done", flush=True)
    else:
        rows, done = [], set()
    t0 = time.time()
    for mech, fn in MECHS.items():
        for frac in MISSING_FRACS:
            for seed in SEEDS_10:
                mtr, mte = fn(X_raw, frac, seed)
                for m in METHODS:
                    if (mech, round(float(frac), 4), int(seed), m) in done:
                        continue
                    X_imp = impute(m, X_raw, mtr, mte, seed=seed,
                                   col_methods=USDA_TRANSFORM_MAP)
                    nr, nr_ref = both_nrmse(X_raw, X_imp, mte, ncol)
                    rows.append(dict(mechanism=mech, missing_frac=frac, seed=seed,
                                     method=m, nrmse=nr, nrmse_refnorm=nr_ref))
                pd.DataFrame(rows).to_csv(OUT, index=False)
            df = pd.DataFrame(rows)
            sub = df[(df.mechanism == mech) & (df.missing_frac == frac)]
            hp = {m: sub[sub.method == m].nrmse.mean() for m in METHODS}
            rp = {m: sub[sub.method == m].nrmse_refnorm.mean() for m in METHODS}
            best = min(hp, key=hp.get)
            print(f"[{mech.upper()} {int(frac*100):3d}%] " +
                  "  ".join(f"{m}={hp[m]:.3f}/{rp[m]:.3f}" for m in METHODS) +
                  f"  BEST(held)={best}  [{(time.time()-t0)/60:.1f}m]", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
