"""
holm_hl.py — Holm-adjusted Wilcoxon + Hodges-Lehmann shift + 95% CI.

Reads sweep_tkpi_10seed.csv (seed-level KNN-k10 vs SoftImpute NRMSE at each rate),
computes per-rate:
  1. Exact two-sided Wilcoxon signed-rank test (n=10 paired masks)
  2. Holm multiple-testing correction across 10 rates
  3. Hodges-Lehmann paired-shift estimate (median of pairwise averages of diffs)
  4. Distribution-free 95% CI for the shift (Walsh averages)

Also refreshes the file from the repinned sweep data (SoftImpute rows were updated
by repin.py).

Output: wilcoxon_per_rate_tkpi.csv
        (missing_frac, knn_mean, si_mean, delta_mean, W, p_unadj, p_holm,
         sig_holm, HL_shift, CI_lo, CI_hi)
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations

from common import EXP_TABLES

IN = EXP_TABLES / "sweep_tkpi_10seed.csv"
OUT = EXP_TABLES / "wilcoxon_per_rate_tkpi.csv"


def hodges_lehmann_ci(diffs, alpha=0.05):
    """
    Hodges-Lehmann estimator and distribution-free CI for paired data.

    The HL estimator is the median of the (n*(n+1)/2) Walsh averages
    (d_i + d_j)/2 for i <= j. The CI is based on the Wilcoxon signed-rank
    order statistics: the K-th smallest and K-th largest Walsh average,
    where K is chosen for the desired confidence level.

    For n=10: the exact 95.1% CI uses K=9 (the 9th and 47th of 55 Walsh averages).
    """
    n = len(diffs)
    # Walsh averages: (d_i + d_j) / 2 for all i <= j
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    walsh = np.sort(walsh)
    hl = float(np.median(walsh))

    # Distribution-free CI: for n=10, exact 95.1% uses K=9
    # General: use scipy to get the critical value, but for n=10 it's well-tabled
    # We use K=9 for n=10 (coverage 95.1%), K=6 for n=8, etc.
    # For safety, compute from the Wilcoxon distribution
    m = len(walsh)  # n*(n+1)/2

    # Table of K values for exact two-sided CI at alpha ~0.05
    # n: K such that P(W >= K) <= alpha/2 under H0
    k_table = {
        5: 1, 6: 2, 7: 4, 8: 6, 9: 8, 10: 9, 11: 11, 12: 14,
        13: 17, 14: 22, 15: 25, 16: 30, 17: 35, 18: 40, 19: 47, 20: 53
    }
    if n in k_table:
        k = k_table[n]
        ci_lo = float(walsh[k - 1])       # K-th smallest (1-indexed -> 0-indexed)
        ci_hi = float(walsh[m - k])        # K-th largest
    else:
        # fallback: use percentile approximation
        ci_lo = float(np.percentile(walsh, 100 * alpha / 2))
        ci_hi = float(np.percentile(walsh, 100 * (1 - alpha / 2)))

    return hl, ci_lo, ci_hi


def holm_correction(pvalues):
    """Holm step-down correction. Returns adjusted p-values."""
    n = len(pvalues)
    idx = np.argsort(pvalues)
    adj = np.zeros(n)
    cummax = 0.0
    for rank, i in enumerate(idx):
        raw = pvalues[i] * (n - rank)
        cummax = max(cummax, raw)
        adj[i] = min(cummax, 1.0)
    return adj


if __name__ == "__main__":
    df = pd.read_csv(IN)
    fracs = sorted(df.missing_frac.unique())
    seeds = sorted(df.seed.unique())
    print(f"TKPI sweep: {len(fracs)} rates, {len(seeds)} seeds", flush=True)

    rows = []
    raw_p = []

    for frac in fracs:
        sub = df[df.missing_frac == frac]
        knn_vals = []
        si_vals = []
        for s in seeds:
            ss = sub[sub.seed == s]
            knn_vals.append(float(ss[ss.method == 'knn_k10'].nrmse.values[0]))
            si_vals.append(float(ss[ss.method == 'softimpute'].nrmse.values[0]))
        knn_arr = np.array(knn_vals)
        si_arr = np.array(si_vals)
        diffs = knn_arr - si_arr  # positive = KNN worse (SI better)

        # Wilcoxon signed-rank, exact, two-sided
        # Note: if all diffs are same sign, W=0 or W=n*(n+1)/2
        stat, p = wilcoxon(knn_arr, si_arr, alternative='two-sided', method='exact')
        raw_p.append(p)

        # Hodges-Lehmann + CI
        hl, ci_lo, ci_hi = hodges_lehmann_ci(diffs)

        rows.append(dict(
            missing_frac=frac,
            knn_mean=float(knn_arr.mean()),
            si_mean=float(si_arr.mean()),
            delta_mean=float(diffs.mean()),
            W=stat,
            p_unadj=p,
            HL_shift=hl,
            CI_lo=ci_lo,
            CI_hi=ci_hi,
        ))

    # Holm correction
    adj_p = holm_correction(np.array(raw_p))
    for i, r in enumerate(rows):
        r['p_holm'] = adj_p[i]
        r['sig_holm'] = adj_p[i] < 0.05

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"\n{'frac':>5}  {'knn':>7}  {'si':>7}  {'delta':>7}  "
          f"{'W':>5}  {'p_raw':>8}  {'p_holm':>8}  {'sig':>4}  "
          f"{'HL':>7}  {'CI_lo':>7}  {'CI_hi':>7}", flush=True)
    for _, r in out.iterrows():
        sig = '***' if r.sig_holm else '   '
        print(f"{r.missing_frac:>5.2f}  {r.knn_mean:>7.4f}  {r.si_mean:>7.4f}  "
              f"{r.delta_mean:>+7.4f}  {r.W:>5.0f}  {r.p_unadj:>8.5f}  "
              f"{r.p_holm:>8.5f}  {sig:>4}  {r.HL_shift:>+7.4f}  "
              f"{r.CI_lo:>+7.4f}  {r.CI_hi:>+7.4f}", flush=True)
    print(f"\n-> {OUT}", flush=True)
