"""
pica.py — PICA baseline, faithful reimplementation (ref [1]).

PICA ("Parameter-based Imputation via Cycle-ensemble Averaging", Vo et al.,
Information Sciences 749 (2026) 123532). No official code shipped with the copy
we have; this follows the paper's Algorithms 1-3 exactly.

Pipeline
--------
DPER  (Alg. 1): direct parameter estimation from incomplete data.
  - mu_i, sigma_ii = mean and *uncorrected* sample variance of feature i's
    OBSERVED values.
  - sigma_ij: using rows where both i,j observed (m complete pairs), with global
    mu_i, mu_j, sigma_ii, sigma_jj, solve the cubic Eq. (1) in sigma_ij and pick
    the real root closest to the case-deletion covariance (s12/m).

DPEI  (Alg. 2): impute a row's missing block by the Gaussian conditional mean
  x_m = mu_m + Sig_mo Sig_oo^{-1} (x_o - mu_o).

CeI   (Alg. 3): cycle-ensemble. Stack the first (k-1) features after feature p
  and slide a window of size k over the p features cyclically. Each window is
  imputed with DPEI using ONLY the features inside the window (local context);
  every feature is imputed k times (once per window containing it) and the
  results are averaged.

Window size k: paper uses k = p when p < 10, else k = 10. TKPI(19)/USDA(18) -> 10.

NOTE for the manuscript: label this a reimplementation from the paper's
description. PICA assumes multivariate normality; the TKPI/USDA nutrient columns
are heavy-tailed with true-zero spikes, so weak performance here is expected and
is itself a reportable finding, not an implementation error.
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DPER — direct parameter estimation (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────
def _dper_pair_sigma(x1, x2, mu1, mu2, s11, s22):
    """
    Solve Eq. (1) for sigma_12 on the complete pairs of features (1,2).
    x1, x2 are the values on rows where BOTH are observed. s11, s22 are the
    global (all-observed) variances of features 1 and 2.
    Returns the real root closest to the case-deletion covariance.
    """
    m = len(x1)
    if m < 2 or s11 <= 0 or s22 <= 0:
        return 0.0
    d1 = x1 - mu1
    d2 = x2 - mu2
    S11 = np.sum(d1 * d1)          # scatter on complete pairs
    S12 = np.sum(d1 * d2)
    S22 = np.sum(d2 * d2)

    # Cubic in t = sigma_12  (from Eq. 1, multiplied through by -1):
    #   m t^3 - S12 t^2 + (S22 s11 + S11 s22 - m s11 s22) t - S12 s11 s22 = 0
    a = m
    b = -S12
    c = (S22 * s11 + S11 * s22 - m * s11 * s22)
    d = -S12 * s11 * s22
    roots = np.roots([a, b, c, d])
    real = roots[np.abs(roots.imag) < 1e-8].real
    if real.size == 0:
        return S12 / m             # fallback: case-deletion covariance
    case_del = S12 / m
    # keep roots that respect the PSD bound |sigma_12| <= sqrt(s11 s22)
    bound = np.sqrt(s11 * s22)
    feasible = real[np.abs(real) <= bound + 1e-9]
    pool = feasible if feasible.size else real
    return float(pool[np.argmin(np.abs(pool - case_del))])


def dper(X):
    """Return (mu, Sigma) estimated directly from X (n,p) with NaN for missing."""
    n, p = X.shape
    obs = ~np.isnan(X)
    mu = np.array([X[obs[:, j], j].mean() if obs[:, j].any() else 0.0
                   for j in range(p)])
    var = np.array([np.mean((X[obs[:, j], j] - mu[j]) ** 2) if obs[:, j].any() else 1e-6
                    for j in range(p)])
    var = np.maximum(var, 1e-12)
    Sigma = np.diag(var).astype(float)
    for i in range(p):
        for j in range(i + 1, p):
            both = obs[:, i] & obs[:, j]
            sij = _dper_pair_sigma(X[both, i], X[both, j],
                                   mu[i], mu[j], var[i], var[j])
            Sigma[i, j] = Sigma[j, i] = sij
    return mu, Sigma


# ─────────────────────────────────────────────────────────────────────────────
# DPEI — conditional-expectation imputation on a feature subset (Algorithm 2)
# ─────────────────────────────────────────────────────────────────────────────
def _dpei_on_subset(Xsub, mu, Sigma, ridge=1e-6):
    """
    Impute NaN in Xsub (n, q) using conditional expectation under (mu, Sigma)
    restricted to those q features. Rows grouped by missingness pattern.
    Returns a copy with the missing entries filled (entries with no observed
    feature in the window are left NaN so the caller can skip them in averaging).
    """
    n, q = Xsub.shape
    obs = ~np.isnan(Xsub)
    out = Xsub.copy()
    patterns = {}
    for r in range(n):
        patterns.setdefault(obs[r].tobytes(), []).append(r)
    for key, idx in patterns.items():
        o = np.frombuffer(key, dtype=bool)
        m = ~o
        if not m.any() or not o.any():
            continue
        idx = np.array(idx)
        oi = np.where(o)[0]
        mi = np.where(m)[0]
        Soo = Sigma[np.ix_(oi, oi)] + ridge * np.eye(oi.size)
        Smo = Sigma[np.ix_(mi, oi)]
        B = Smo @ np.linalg.inv(Soo)
        Xo = Xsub[np.ix_(idx, oi)]
        out[np.ix_(idx, mi)] = mu[mi] + (Xo - mu[oi]) @ B.T
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CeI + PICA (Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────
def pica_impute(X_input_norm, k=None):
    """
    PICA imputation in normalised space. X_input_norm is (n,p) with NaN missing.
    k = window size (default: p if p < 10 else 10). Returns imputed (n,p) matrix
    clipped to [0,1]. Deterministic (no random seed needed).
    """
    X = X_input_norm.astype(float)
    n, p = X.shape
    if k is None:
        k = p if p < 10 else 10
    k = min(k, p)

    mu, Sigma = dper(X)

    sums = np.zeros((n, p))
    counts = np.zeros((n, p))
    feat_order = list(range(p))
    for start in range(p):                      # p cyclic windows of size k
        win = [feat_order[(start + t) % p] for t in range(k)]
        Xsub = X[:, win]
        mu_w = mu[win]
        Sig_w = Sigma[np.ix_(win, win)]
        imp = _dpei_on_subset(Xsub, mu_w, Sig_w)
        for local, jg in enumerate(win):
            col = imp[:, local]
            filled = ~np.isnan(col)
            sums[filled, jg] += col[filled]
            counts[filled, jg] += 1

    out = X.copy()
    miss = np.isnan(X)
    have = miss & (counts > 0)
    out[have] = sums[have] / counts[have]
    # any missing entry never imputed by any window -> fall back to feature mean
    still = miss & (counts == 0)
    if still.any():
        out[still] = np.take(mu, np.where(still)[1])
    return np.clip(out, 0.0, 1.0)


if __name__ == "__main__":
    import time
    from common import load_tkpi, make_random_mask, _prep_norm
    from utils import evaluate_imputation
    X_raw, _, cols = load_tkpi()
    mtr, mte = make_random_mask(X_raw, 0.20, 123)
    prep, Xin = _prep_norm(X_raw, mtr, mte)
    t = time.time()
    Xn = pica_impute(Xin)
    Ximp = prep.inverse_transform(Xn)
    v = evaluate_imputation(X_raw, Ximp, mte, cols)['median_nrmse']
    print(f"PICA @20%/seed123: NRMSE={v:.4f}  (k=10, {time.time()-t:.1f}s)")
