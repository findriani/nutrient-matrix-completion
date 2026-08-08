# Mask archive — reproducibility manifest

All masks are generated deterministically from a seed + missing fraction + mechanism.
Re-running the mask function with these parameters reproduces the exact same mask.

## Seeds
```
SEEDS_10 = [123, 456, 789, 1024, 1337, 2024, 7, 99, 555, 31337]
```

## Missing fractions
```
MISSING_FRACS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
```

## Mask functions (in `experiments/common.py` and `experiments/usda.py`)

### TKPI — `make_random_mask(X, frac, seed)`
- MCAR: `rng = np.random.RandomState(seed); mask = rng.rand(*X.shape) < frac`
- Split: 50% train / 50% test among masked cells
- Applied to: sweep_tkpi, main_scenarioA, baselines_scenarioA, baselines_sweep,
  per_nutrient_A, pica_scenarioA, pica_sweep

### USDA — `make_mcar_mask(X, frac, seed)`
- Same MCAR logic on the USDA matrix
- Applied to: sweep_usda, downstream_usda, pica_usda_mcar

### USDA MAR — `make_mar_mask(X, frac, seed)`
- Energy-driven: masking probability rises with observed Energy column
- Applied to: sweep_usda_mechanisms (mechanism="mar")

### USDA MNAR — `make_mnar_mask(X, frac, seed)`
- Low-value self-masking (LOD-style censoring)
- Applied to: sweep_usda_mechanisms (mechanism="mnar")

### TKPI Scenario B — `block_micro_mask(X, food_frac, seed)`
- Block missingness at food_frac=0.30
- Applied to: main_scenarioB

## Environment
See `environment_freeze.txt` for the exact pip freeze.
Python version embedded at end of that file.

## SoftImpute determinism
SoftImpute's `randomized_svd` is pinned to `random_state=0` via monkeypatch
in `experiments/compat_patch.py`. IterativeSVD's `TruncatedSVD` is similarly pinned.
Verified: repeated runs produce bit-identical output (max delta = 0.000).
