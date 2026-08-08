"""
holm_hl_pica.py -- Holm-adjusted Wilcoxon + Hodges-Lehmann shift for
PICA vs SoftImpute on the TKPI additional-masking sweep. Tests PICA's role
statistically rather than just describing the numbers.

Post-hoc analysis on already-collected per-seed sweep data
(sweep_tkpi_10seed.csv, pica_sweep_10seed.csv) -- no new imputation runs.
Reuses the exact methodology of holm_hl.py (KNN vs SoftImpute) so the
two tests are directly comparable.

Output: wilcoxon_per_rate_tkpi_pica.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import EXP_TABLES
from holm_hl import hodges_lehmann_ci, holm_correction

SI = EXP_TABLES / "sweep_tkpi_10seed.csv"
PICA = EXP_TABLES / "pica_sweep_10seed.csv"
OUT = EXP_TABLES / "wilcoxon_per_rate_tkpi_pica.csv"

if __name__ == "__main__":
    si_df = pd.read_csv(SI)
    si_df = si_df[si_df.method == "softimpute"]
    pica_df = pd.read_csv(PICA)

    fracs = sorted(set(si_df.missing_frac.unique()) & set(pica_df.missing_frac.unique()))
    rows = []
    raw_p = []

    for frac in fracs:
        si_sub = si_df[si_df.missing_frac == frac].set_index("seed")["nrmse"]
        pica_sub = pica_df[pica_df.missing_frac == frac].set_index("seed")["nrmse"]
        seeds = sorted(set(si_sub.index) & set(pica_sub.index))
        assert len(seeds) == 10, f"expected 10 paired seeds at {frac}, got {len(seeds)}"
        si_arr = np.array([si_sub[s] for s in seeds])
        pica_arr = np.array([pica_sub[s] for s in seeds])
        diffs = pica_arr - si_arr  # positive = PICA worse (SI better); negative = PICA better

        stat, p = wilcoxon(pica_arr, si_arr, alternative="two-sided", method="exact")
        raw_p.append(p)
        hl, ci_lo, ci_hi = hodges_lehmann_ci(diffs)

        rows.append(dict(
            missing_frac=frac,
            pica_mean=float(pica_arr.mean()),
            si_mean=float(si_arr.mean()),
            delta_mean=float(diffs.mean()),
            W=stat, p_unadj=p,
            HL_shift=hl, CI_lo=ci_lo, CI_hi=ci_hi,
        ))

    adj_p = holm_correction(np.array(raw_p))
    for i, r in enumerate(rows):
        r["p_holm"] = adj_p[i]
        r["sig_holm"] = bool(adj_p[i] < 0.05)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"{'frac':>5}  {'pica':>7}  {'si':>7}  {'delta':>7}  {'p_holm':>8}  {'sig':>4}  "
          f"{'HL':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for _, r in out.iterrows():
        sig = "*" if r.sig_holm else " "
        print(f"{r.missing_frac:>5.2f}  {r.pica_mean:>7.4f}  {r.si_mean:>7.4f}  "
              f"{r.delta_mean:>+7.4f}  {r.p_holm:>8.5f}  {sig:>4}  "
              f"{r.HL_shift:>+7.4f}  {r.CI_lo:>+7.4f}  {r.CI_hi:>+7.4f}")
    print(f"-> {OUT}")
