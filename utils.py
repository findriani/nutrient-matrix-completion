"""
utils.py — Shared utilities for TKPI Matrix Completion pipeline.

Provides:
  NutrientPreprocessor  — leakage-safe per-column power-transform + MinMaxScaler
  evaluate_imputation   — standardized evaluation on held-out entries only

Supported transform methods:
  'yeo-johnson'  — PowerTransformer(yeo-johnson); handles zeros & negatives.
                   Best for macro nutrients (Energy, Protein, Fat, Carb, Fiber).
  'box-cox'      — PowerTransformer(box-cox); requires positive values (zeros
                   are shifted by 1e-6 automatically).
  'log1p'        — Fixed log(1+x) transform; stable for sparse micronutrients.
  'none'         — No power transform; MinMaxScaler only.

Per-column transforms are supported via col_methods= parameter.
Use config.NUTRIENT_TRANSFORM_MAP for the recommended macro/micro split.
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PowerTransformer


# ─────────────────────────────────────────────────────────────────────────────
class NutrientPreprocessor:
    """
    Leakage-safe per-column power-transform + MinMaxScaler.

    The power transform is fitted exclusively on observed training entries so
    no information from held-out or missing cells leaks into normalization.

    Usage (inside each masking fold):
        from config import NUTRIENT_TRANSFORM_MAP
        prep = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
        prep.fit(X_raw, mask_train)
        X_norm  = prep.transform(X_raw)
        X_back  = prep.inverse_transform(X_norm_imputed)

    Parameters
    ----------
    method : str
        Single method for all columns: 'yeo-johnson' (default), 'box-cox',
        or 'log1p'. Ignored when col_methods is provided.
    col_methods : list of str, optional
        Per-column method list of length n_cols. Overrides method when given.
        Use config.NUTRIENT_TRANSFORM_MAP for the macro/micro split.
    """

    VALID_METHODS = ('yeo-johnson', 'box-cox', 'log1p', 'none')

    def __init__(self, method: str = 'none', col_methods: list = None, scale: bool = True):
        if col_methods is None:
            if method not in self.VALID_METHODS:
                raise ValueError(
                    f"method must be one of {self.VALID_METHODS}, got {method!r}"
                )
        else:
            for m in col_methods:
                if m not in self.VALID_METHODS:
                    raise ValueError(
                        f"col_methods contains invalid method {m!r}; "
                        f"must be one of {self.VALID_METHODS}"
                    )
        self.method      = method
        self.col_methods = col_methods            # per-column override or None
        self.scale       = scale                  # if False, skip MinMaxScaler
        self.scalers_: dict      = {}
        self.transformers_: dict = {}   # PowerTransformer per column (None for log1p)
        self._n_cols: int        = 0

    # ------------------------------------------------------------------
    def _col_method(self, j: int) -> str:
        """Return the transform method for column j."""
        return self.col_methods[j] if self.col_methods is not None else self.method

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, observed_mask: np.ndarray) -> "NutrientPreprocessor":
        """
        Fit per-column transformer + scaler on entries where
        observed_mask == True AND value is not NaN.
        """
        self._n_cols = X.shape[1]
        self.scalers_      = {}
        self.transformers_ = {}

        for j in range(self._n_cols):
            col  = X[:, j]
            mask = observed_mask[:, j] & ~np.isnan(col)
            vals = col[mask]

            if len(vals) == 0:
                # fallback: identity scaler for all-NaN columns
                self.transformers_[j] = None
                sc = MinMaxScaler()
                sc.fit(np.array([[0.0], [1.0]]))
                self.scalers_[j] = sc
                continue

            # ── apply power transform ─────────────────────────────────
            m = self._col_method(j)

            if m == 'none':
                vals_t = vals.reshape(-1, 1)
                self.transformers_[j] = None

            elif m == 'log1p':
                vals_t = np.log1p(np.maximum(vals, 0.0)).reshape(-1, 1)
                self.transformers_[j] = None

            elif m == 'yeo-johnson':
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                pt.fit(vals.reshape(-1, 1))
                vals_t = pt.transform(vals.reshape(-1, 1))
                self.transformers_[j] = pt

            else:  # box-cox — requires strictly positive input
                pt = PowerTransformer(method='box-cox', standardize=False)
                vals_pos = np.maximum(vals, 1e-6).reshape(-1, 1)
                pt.fit(vals_pos)
                vals_t = pt.transform(vals_pos)
                self.transformers_[j] = pt

            # ── fit MinMaxScaler on transformed values ────────────────
            if self.scale:
                sc = MinMaxScaler()
                sc.fit(vals_t)
                self.scalers_[j] = sc

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply power transform + MinMaxScaling column-wise.
        NaN values are passed through unchanged.
        """
        X_out = np.full_like(X, np.nan, dtype=float)

        for j in range(X.shape[1]):
            col   = X[:, j]
            valid = ~np.isnan(col)
            if not valid.any():
                continue

            vals = col[valid]

            if self._col_method(j) == 'none':
                vals_t = vals.reshape(-1, 1)
            elif self._col_method(j) == 'log1p' or self.transformers_[j] is None:
                # None means the column had no training data (e.g. cold-start);
                # fall back to log1p so the operation is still defined.
                vals_t = np.log1p(np.maximum(vals, 0.0)).reshape(-1, 1)
            elif self._col_method(j) == 'yeo-johnson':
                vals_t = self.transformers_[j].transform(vals.reshape(-1, 1))
            else:  # box-cox
                vals_t = self.transformers_[j].transform(
                    np.maximum(vals, 1e-6).reshape(-1, 1)
                )

            if self.scale:
                X_out[valid, j] = self.scalers_[j].transform(vals_t).ravel()
            else:
                X_out[valid, j] = vals_t.ravel()

        return X_out

    # ------------------------------------------------------------------
    def inverse_transform(self, X_imputed: np.ndarray) -> np.ndarray:
        """
        Inverse MinMaxScaling + inverse power transform.
        Clips result to [0, ∞) for physical validity.
        NaN values are passed through.
        """
        X_out = np.full_like(X_imputed, np.nan, dtype=float)

        for j in range(X_imputed.shape[1]):
            col   = X_imputed[:, j]
            valid = ~np.isnan(col)
            if not valid.any():
                continue

            if self.scale:
                # inverse MinMaxScaler
                inv = self.scalers_[j].inverse_transform(
                    col[valid].reshape(-1, 1)
                )
                # Clip to training range to prevent NaN in power-transform inversion.
                # Matrix completion (SoftImpute/NMF) can extrapolate outside [0,1];
                # after inverse MinMaxScaling those values fall outside the domain
                # that PowerTransformer.inverse_transform() was fitted on, causing NaN.
                inv = np.clip(inv,
                              self.scalers_[j].data_min_,
                              self.scalers_[j].data_max_)
            else:
                inv = col[valid].reshape(-1, 1)

            # inverse power transform
            if self._col_method(j) == 'none':
                X_out[valid, j] = inv.ravel()
            elif self._col_method(j) == 'log1p' or self.transformers_[j] is None:
                # None = cold-start column with no training data; log1p was
                # used as the fallback in transform(), so invert with expm1.
                X_out[valid, j] = np.expm1(inv.ravel())
            else:
                X_out[valid, j] = self.transformers_[j].inverse_transform(inv).ravel()

        return np.clip(X_out, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_imputation(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask_test: np.ndarray,
    nutrient_names: list,
) -> dict:
    """
    Evaluate imputation quality exclusively on held-out entries.

    Parameters
    ----------
    X_true        : (n, p) ground-truth matrix in original scale.
    X_imputed     : (n, p) imputed matrix in original scale.
    mask_test     : (n, p) bool; True where entry was held out for evaluation.
    nutrient_names: list of p column names.

    Returns
    -------
    dict:
        per_nutrient  → {name: {nrmse, mae, ks_stat, ks_p, n_eval}}
        median_nrmse  → float  (median across nutrients with ≥1 held-out entry)
        validity_rate → float  (fraction of imputed held-out values ≥ 0)
    """
    results: dict = {
        "per_nutrient": {},
        "median_nrmse": np.nan,
        "validity_rate": np.nan,
    }

    nrmse_list = []

    for j, name in enumerate(nutrient_names):
        idx = mask_test[:, j]
        if idx.sum() == 0:
            continue

        y_true = X_true[idx, j]
        y_pred = X_imputed[idx, j]

        # NRMSE normalised by std(y_true) — robust to outliers vs range
        std_true = np.std(y_true)
        nrmse    = np.sqrt(np.mean((y_true - y_pred) ** 2)) / (std_true + 1e-10)
        mae      = np.mean(np.abs(y_true - y_pred))
        ks_stat, ks_p = stats.ks_2samp(y_true, y_pred)

        results["per_nutrient"][name] = {
            "nrmse":   nrmse,
            "mae":     mae,
            "ks_stat": ks_stat,
            "ks_p":    ks_p,
            "n_eval":  int(idx.sum()),
        }
        nrmse_list.append(nrmse)

    if nrmse_list:
        results["median_nrmse"] = float(np.median(nrmse_list))

    held_out_vals = X_imputed[mask_test]
    if len(held_out_vals) > 0:
        results["validity_rate"] = float((held_out_vals >= 0).mean())

    return results
