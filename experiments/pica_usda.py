"""
pica_usda.py — PICA sweep on USDA MCAR (PICA's intended regime).

Sweeps PICA across MISSING_FRACS x SEEDS_10 on the USDA SR Legacy complete-case
subset under MCAR masking. PICA assumes multivariate normality and MCAR — this is
its best-case scenario.

Output: pica_usda_mcar_10seed.csv (missing_frac, seed, method, nrmse)
"""
import time
import numpy as np
import pandas as pd

from usda import load_usda, make_mcar_mask, USDA_TRANSFORM_MAP
from common import _prep_norm, MISSING_FRACS, SEEDS_10, EXP_TABLES
from pica import pica_impute

from _bootstrap import PROJECT_ROOT  # noqa: F401
from utils import evaluate_imputation

OUT = EXP_TABLES / "pica_usda_mcar_10seed.csv"


if __name__ == "__main__":
    t0 = time.time()
    X_raw, _, cols = load_usda()
    ncol = len(cols)
    print(f"USDA {X_raw.shape}  PICA sweep  seeds={SEEDS_10}", flush=True)

    # resume-safe
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
        done = {(round(float(r["missing_frac"]), 4), int(r["seed"])) for r in rows}
        print(f"Resuming: {len(done)} rows already done", flush=True)
    else:
        rows, done = [], set()

    for frac in MISSING_FRACS:
        for seed in SEEDS_10:
            if (round(float(frac), 4), int(seed)) in done:
                continue
            mtr, mte = make_mcar_mask(X_raw, frac, seed)
            prep, Xin = _prep_norm(X_raw, mtr, mte, col_methods=USDA_TRANSFORM_MAP)
            Xn = pica_impute(Xin)
            X_imp = prep.inverse_transform(Xn)
            nrmse = evaluate_imputation(X_raw, X_imp, mte, cols)['median_nrmse']
            rows.append(dict(missing_frac=frac, seed=seed, method='pica', nrmse=nrmse))
            print(f"  frac={frac:.2f} seed={seed} NRMSE={nrmse:.4f}  "
                  f"[{(time.time()-t0)/60:.1f}m]", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    for frac in MISSING_FRACS:
        sub = df[df.missing_frac == frac]
        print(f"  {int(frac*100):3d}%  mean={sub.nrmse.mean():.4f}  "
              f"sd={sub.nrmse.std():.4f}", flush=True)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
