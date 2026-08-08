"""
compat_patch.py — sklearn>=1.6 / fancyimpute 0.7.0 compatibility shim.

fancyimpute 0.7.0 calls ``check_array(X, force_all_finite=False)``. In
scikit-learn 1.6 that keyword was renamed to ``ensure_all_finite`` and in a
later release the old name is removed, raising::

    TypeError: check_array() got an unexpected keyword argument 'force_all_finite'

This module wraps ``sklearn.utils.check_array`` (and the copy in
``sklearn.utils.validation``) so that ``force_all_finite`` is transparently
translated to ``ensure_all_finite``. Import this module *before* importing
fancyimpute so the wrapped function is bound when fancyimpute's submodules do
``from sklearn.utils import check_array``.

Usage
-----
    import compat_patch  # noqa: F401  (must precede fancyimpute import)
    from fancyimpute import SoftImpute

The SoftImpute algorithm itself is untouched; only the input-validation
call signature is adapted.
"""

import functools
import sklearn.utils
import sklearn.utils.validation as _skv


def _wrap(orig):
    @functools.wraps(orig)
    def check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return orig(*args, **kwargs)
    return check_array


# Patch both the canonical location and the re-export used by fancyimpute.
_patched = _wrap(_skv.check_array)
_skv.check_array = _patched
sklearn.utils.check_array = _patched


# ═════════════════════════════════════════════════════════════════════════════
# Deterministic spectral decompositions (for exact run-to-run reproducibility)
# ═════════════════════════════════════════════════════════════════════════════
# fancyimpute's SoftImpute calls ``randomized_svd(..., random_state=None)`` and
# IterativeSVD builds ``TruncatedSVD(...)`` with no random_state, so both draw a
# fresh RNG on every call and vary run-to-run (observed at up to ~0.02 NRMSE).
# Every OTHER method in the pipeline already fixes its seed. We pin the seed here
# so SoftImpute / IterativeSVD become deterministic and all reported numbers are
# exactly reproducible. This must run *before* fancyimpute is imported elsewhere;
# common.py imports this module first, so importing the submodules here is safe
# (check_array is already patched above).
SVD_SEED = 0


def _seed_randomized_svd(orig):
    @functools.wraps(orig)
    def randomized_svd(*args, **kwargs):
        # SoftImpute passes random_state=None explicitly; force a fixed seed.
        if kwargs.get("random_state", None) is None:
            kwargs["random_state"] = SVD_SEED
        return orig(*args, **kwargs)
    return randomized_svd


def _seed_truncated_svd(orig_cls):
    @functools.wraps(orig_cls, assigned=("__doc__",))
    def TruncatedSVD(*args, **kwargs):
        # IterativeSVD omits random_state -> default None -> global RNG.
        kwargs.setdefault("random_state", SVD_SEED)
        return orig_cls(*args, **kwargs)
    return TruncatedSVD


import fancyimpute.soft_impute as _si          # noqa: E402
import fancyimpute.iterative_svd as _isvd      # noqa: E402

_si.randomized_svd = _seed_randomized_svd(_si.randomized_svd)
_isvd.TruncatedSVD = _seed_truncated_svd(_isvd.TruncatedSVD)

__all__ = ["SVD_SEED"]
