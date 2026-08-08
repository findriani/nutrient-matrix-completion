"""
pica_run.py — evaluate the PICA baseline on TKPI (ref [1]).

Runs PICA (pica.pica_impute) on the same leakage-safe MinMax pipeline as
every other method: Scenario A (20%, 10 seeds) and the missing-rate sweep
(5-50%, 10 seeds). Resumable. Outputs:
  outputs/tables/pica_scenarioA_10seed.csv
  outputs/tables/pica_sweep_10seed.csv
"""
import time
import numpy as np
import pandas as pd

from common import (
    load_tkpi, make_random_mask, _prep_norm, MISSING_FRACS, SEEDS_10, EXP_TABLES,
)
from pica import pica_impute
from utils import evaluate_imputation

X_raw, _, cols = load_tkpi()


def run_pica(mtr, mte):
    prep, Xin = _prep_norm(X_raw, mtr, mte)
    Xn = pica_impute(Xin)
    Ximp = prep.inverse_transform(Xn)
    return evaluate_imputation(X_raw, Ximp, mte, cols)['median_nrmse']


if __name__ == "__main__":
    t0 = time.time()

    # ── Scenario A (20%) ──────────────────────────────────────────────────
    fA = EXP_TABLES / "pica_scenarioA_10seed.csv"
    rowsA, doneA = [], set()
    if fA.exists():
        rowsA = pd.read_csv(fA).to_dict('records')
        doneA = {int(r['seed']) for r in rowsA}
    for seed in SEEDS_10:
        if seed in doneA:
            continue
        mtr, mte = make_random_mask(X_raw, 0.20, seed)
        rowsA.append(dict(seed=seed, method='pica', nrmse=run_pica(mtr, mte)))
        pd.DataFrame(rowsA).to_csv(fA, index=False)
        print(f"  [A] seed {seed} done  [{(time.time()-t0)/60:.1f}m]", flush=True)
    dA = pd.DataFrame(rowsA)
    print(f"Scenario A PICA: NRMSE={dA.nrmse.mean():.4f} ± {dA.nrmse.std():.4f}")

    # ── Sweep (5-50%) ─────────────────────────────────────────────────────
    fS = EXP_TABLES / "pica_sweep_10seed.csv"
    rows, done = [], set()
    if fS.exists():
        rows = pd.read_csv(fS).to_dict('records')
        done = {(round(r['missing_frac'], 3), int(r['seed'])) for r in rows}
    for frac in MISSING_FRACS:
        for seed in SEEDS_10:
            if (round(frac, 3), seed) in done:
                continue
            mtr, mte = make_random_mask(X_raw, frac, seed)
            rows.append(dict(missing_frac=frac, seed=seed, method='pica',
                             nrmse=run_pica(mtr, mte)))
            pd.DataFrame(rows).to_csv(fS, index=False)
        sub = pd.DataFrame(rows)
        m = sub[sub.missing_frac == frac].nrmse.mean()
        print(f"[{int(frac*100):3d}%] pica={m:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
