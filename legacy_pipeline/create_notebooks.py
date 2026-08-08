"""
create_notebooks.py
Generates all 6 Jupyter notebooks for the TKPI Matrix Completion pipeline.
Run once: python create_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent


def cell(source: str, cell_type: str = "code") -> nbf.NotebookNode:
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source)
    c = nbf.v4.new_code_cell(source)
    return c


def notebook(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9.0"}
    nb.cells = cells
    return nb


# ── Colab setup cell (prepended to every notebook) ───────────────────────────
def colab_setup_cell(extra_packages: list = None) -> nbf.NotebookNode:
    """
    Returns a setup cell that:
      - Detects Google Colab
      - Installs packages missing from Colab's default environment
      - Reminds the user which files to upload
      - Optionally mounts Google Drive (commented out by default)
    Safe to run locally: all Colab-specific steps are skipped.
    """
    pkgs = ["missingno", "fancyimpute"] + (extra_packages or [])
    # Build the list literal as it will appear in the generated cell source,
    # e.g. ["missingno", "fancyimpute"] so pip receives them as separate args.
    pkgs_literal = str(pkgs)

    src = f'''\
# ─── COLAB SETUP (safe to run locally — all steps are skipped outside Colab) ─
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import subprocess, sys
    from pathlib import Path

    # ── 1. Install packages not bundled with Colab ────────────────────────
    _pkgs = {pkgs_literal}
    print(f"Installing: {{', '.join(_pkgs)}} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + _pkgs + ["-q"],
        check=True,
    )
    print("Packages installed [OK]")

    # ── 2. Upload reminder ────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("UPLOAD REQUIRED — Files panel (folder icon, left sidebar)")
    print("  Upload these three files to /content/ :")
    print("    TKPIv2.csv    config.py    utils.py")
    print("-" * 60)
    print()

    # ── 3. Mount Google Drive ─────────────────────────────────────────────
    # All outputs (figures, tables, .pkl results) are saved here so they
    # persist across sessions and are available to every notebook.
    from google.colab import drive
    drive.mount("/content/drive")

    # Change this path if you want a different Drive folder:
    DRIVE_OUTPUT = "/content/drive/MyDrive/Penelitian/MatrixFactorization3/outputs"

    sys.path.insert(0, "/content")
    import config
    config.OUTPUT_DIR  = Path(DRIVE_OUTPUT)
    config.FIGURES_DIR = Path(DRIVE_OUTPUT) / "figures"
    config.TABLES_DIR  = Path(DRIVE_OUTPUT) / "tables"
    config.RESULTS_DIR = Path(DRIVE_OUTPUT) / "results"
    for d in [config.FIGURES_DIR, config.TABLES_DIR, config.RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Outputs will be saved to: {{DRIVE_OUTPUT}}")
else:
    print("Running locally [OK]")
'''
    return cell(src)


# ══════════════════════════════════════════════════════════════════════════════
# 01_eda.ipynb
# ══════════════════════════════════════════════════════════════════════════════
def make_01_eda():
    cells = [
        cell("# 01 — Exploratory Data Analysis\n"
             "Audit shape, missingness, zero-rate, singular value spectrum, "
             "and per-nutrient distributions.", "markdown"),

        colab_setup_cell(),   # ← Colab setup (installs missingno + fancyimpute)

        cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_PATH, FIGURES_DIR, TABLES_DIR, RESULTS_DIR,
    ID_COL, NUTRIENT_COLS, MACRO_COLS, MICRO_COLS,
    COLUMN_RENAME, TREAT_ZEROS_AS_OBSERVED,
)

for d in [FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11})
print("Config loaded. DATA_PATH:", DATA_PATH)
"""),

        cell("## 1. Load & Audit", "markdown"),

        cell("""\
df = pd.read_csv(DATA_PATH)
df.rename(columns=COLUMN_RENAME, inplace=True)   # Indonesian → English
print(f"Shape: {df.shape}")
print(f"\\nColumn dtypes:\\n{df.dtypes.to_string()}")
print(f"\\nFirst 3 rows:")
df.head(3)
"""),

        cell("""\
# Sentinel / unexpected value check
nutrient_df = df[NUTRIENT_COLS].copy().astype(float)
print("NaN count per nutrient:")
print(nutrient_df.isnull().sum().to_string())
print("\\nNegative values per nutrient:")
print((nutrient_df < 0).sum().to_string())
"""),

        cell("## 2. Zero vs NaN Disambiguation", "markdown"),

        cell("""\
missing_rate = nutrient_df.isnull().mean() * 100
zero_rate    = (nutrient_df == 0).sum() / len(nutrient_df) * 100

summary = pd.DataFrame({
    'observed_pct': (100 - missing_rate).round(2),
    'missing_pct':  missing_rate.round(2),
    'zero_pct':     zero_rate.round(2),
    'mean':         nutrient_df.mean().round(4),
    'std':          nutrient_df.std().round(4),
    'min':          nutrient_df.min(),
    'max':          nutrient_df.max(),
}, index=NUTRIENT_COLS)

print(summary.to_string())
"""),

        cell("""\
fig, ax = plt.subplots(figsize=(13, 5))
x     = np.arange(len(NUTRIENT_COLS))
width = 0.4

ax.bar(x - width/2, missing_rate.values, width,
       label='Missing (NaN) %', color='tomato', alpha=0.85)
ax.bar(x + width/2, zero_rate.values, width,
       label='Zero %', color='steelblue', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(NUTRIENT_COLS, rotation=45, ha='right')
ax.set_ylabel('Percentage (%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'zero_vs_nan_rate.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"TREAT_ZEROS_AS_OBSERVED = {TREAT_ZEROS_AS_OBSERVED}")
"""),

        cell("## 3. Missingness Patterns (MCAR vs MNAR evidence)", "markdown"),

        cell("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
msno.matrix(nutrient_df, ax=axes[0], fontsize=9, sparkline=False)
msno.heatmap(nutrient_df, ax=axes[1], fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'missingness_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
"""),

        cell("""\
fig, ax = plt.subplots(figsize=(10, 6))
msno.dendrogram(nutrient_df, ax=ax, fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'missingness_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()
"""),

        cell("## 4. Singular Value Spectrum — H1 Evidence", "markdown"),

        cell("""\
# SVD on complete rows after z-score normalization
complete_mask = ~nutrient_df.isnull().any(axis=1)
X_complete    = nutrient_df[complete_mask].values.astype(float)
print(f"Complete rows for SVD: {X_complete.shape[0]} / {len(nutrient_df)}")

X_z = (X_complete - X_complete.mean(axis=0)) / (X_complete.std(axis=0) + 1e-10)
U, s, Vt = svd(X_z, full_matrices=False)

cum_var = np.cumsum(s**2) / np.sum(s**2) * 100
k80 = int(np.argmax(cum_var >= 80)) + 1
k90 = int(np.argmax(cum_var >= 90)) + 1
print(f"Rank for 80% variance: {k80}")
print(f"Rank for 90% variance: {k90}")
print(f"Top-3 explain: {cum_var[2]:.1f}%")
print(f"Top-6 explain: {cum_var[5]:.1f}%")
"""),

        cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Scree
axes[0].bar(range(1, len(s)+1), s**2 / s[0]**2,
            color='steelblue', alpha=0.75)
axes[0].plot(range(1, len(s)+1), s**2 / s[0]**2,
             'o-', color='navy', markersize=4)
axes[0].set_xlabel('Singular Value Index')
axes[0].set_ylabel('Normalized Eigenvalue (σ²/σ₁²)')
axes[0].set_xlim([0.5, len(s) + 0.5])
axes[0].grid(axis='y', alpha=0.3)

# Right: Cumulative variance
axes[1].plot(range(1, len(s)+1), cum_var, 'ro-', markersize=5)
axes[1].axhline(y=80, color='blue',  linestyle='--', alpha=0.6, label='80%')
axes[1].axhline(y=90, color='green', linestyle='--', alpha=0.6, label='90%')
axes[1].axvline(x=k80, color='blue',  linestyle=':', alpha=0.5)
axes[1].axvline(x=k90, color='green', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].legend(fontsize=9)
axes[1].set_xlim([0.5, len(s) + 0.5])
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'singular_value_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"H1 indicator: top {k90} components explain 90% of variance.")
"""),

        cell("## 5. Per-Nutrient Distributions (raw vs log1p)", "markdown"),

        cell("""\
ncols = 5
nrows = (len(NUTRIENT_COLS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.ravel()

for i, col in enumerate(NUTRIENT_COLS):
    vals = nutrient_df[col].dropna().values
    ax   = axes[i]
    ax.hist(vals,           bins=40, alpha=0.5, color='steelblue',
            label='raw', density=True)
    ax.hist(np.log1p(vals), bins=40, alpha=0.5, color='tomato',
            label='log1p', density=True)
    ax.set_xlabel(col, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distributions_log1p.png', dpi=300, bbox_inches='tight')
plt.show()
"""),

        cell("## 6. Save Nutrient Summary Table", "markdown"),

        cell("""\
summary.to_csv(TABLES_DIR / 'nutrient_summary.csv')
print("Saved:", TABLES_DIR / 'nutrient_summary.csv')
print(summary.round(3).to_string())
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# 02_preprocessing.ipynb
# ══════════════════════════════════════════════════════════════════════════════
def make_02_preprocessing():
    cells = [
        cell("# 02 — Preprocessing\n"
             "Demonstrate leakage-safe `NutrientPreprocessor` and serialize "
             "the canonical raw matrix.", "markdown"),

        colab_setup_cell(),   # ← Colab setup

        cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import joblib

from config import (
    DATA_PATH, RESULTS_DIR, ID_COL, NUTRIENT_COLS, COLUMN_RENAME,
    MACRO_TRANSFORM, MICRO_TRANSFORM, NUTRIENT_TRANSFORM_MAP,
)
from utils import NutrientPreprocessor

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Imports OK  |  macro={MACRO_TRANSFORM!r}  micro={MICRO_TRANSFORM!r}")
"""),

        cell("## 1. Load Raw Matrix", "markdown"),

        cell("""\
df     = pd.read_csv(DATA_PATH)
df.rename(columns=COLUMN_RENAME, inplace=True)   # Indonesian → English
X_raw  = df[NUTRIENT_COLS].values.astype(float)   # (1146, 19)
food_ids = df[ID_COL].values

print(f"X_raw shape : {X_raw.shape}")
print(f"Total NaN   : {np.isnan(X_raw).sum()}")
print(f"Total finite: {np.isfinite(X_raw).sum()}")
"""),

        cell("## 2. Demonstrate NutrientPreprocessor (leakage-safe)", "markdown"),

        cell("""\
# Simulate a full-data scenario: all observed entries are 'training'
observed_mask = ~np.isnan(X_raw)

prep = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
prep.fit(X_raw, observed_mask)

X_norm = prep.transform(X_raw)

print("Normalized column stats (observed entries only):")
print(f"{'Nutrient':<14}  {'min':>8}  {'max':>8}  {'mean':>8}")
for j, col in enumerate(NUTRIENT_COLS):
    vals = X_norm[observed_mask[:, j], j]
    if len(vals):
        print(f"{col:<14}  {vals.min():8.4f}  {vals.max():8.4f}  {vals.mean():8.4f}")
"""),

        cell("## 3. Round-Trip Accuracy Test", "markdown"),

        cell("""\
X_back = prep.inverse_transform(X_norm)

print("Round-trip max absolute error per nutrient:")
print(f"{'Nutrient':<14}  {'MaxAbsErr':>12}")
ok = True
for j, col in enumerate(NUTRIENT_COLS):
    mask  = observed_mask[:, j]
    err   = np.abs(X_raw[mask, j] - X_back[mask, j]).max() if mask.any() else 0.0
    flag  = "  ✓" if err < 1e-3 else "  ← LARGE"
    if err >= 1e-3:
        ok = False
    print(f"{col:<14}  {err:12.6f}{flag}")

print("\\nRound-trip OK:", ok)
"""),

        cell("## 4. Transform Comparison (log1p vs yeo-johnson vs box-cox)", "markdown"),

        cell("""\
# Compare normality (skewness) of each transform on the full observed data
# Lower |skewness| after transform = better normalization of that nutrient
from scipy.stats import skew

transforms = ['log1p', 'yeo-johnson', 'box-cox']
skew_results = {t: [] for t in transforms}

for method in transforms:
    prep_cmp = NutrientPreprocessor(method=method)
    prep_cmp.fit(X_raw, observed_mask)
    X_t = prep_cmp.transform(X_raw)
    for j in range(X_t.shape[1]):
        vals = X_t[observed_mask[:, j], j]
        skew_results[method].append(abs(skew(vals)) if len(vals) > 3 else np.nan)

skew_df = pd.DataFrame(skew_results, index=NUTRIENT_COLS)
print("Absolute skewness after transform (lower = more normal):")
print(skew_df.round(3).to_string())
print(f"\\nMean |skewness|:  log1p={skew_df['log1p'].mean():.3f}  "
      f"yeo-johnson={skew_df['yeo-johnson'].mean():.3f}  "
      f"box-cox={skew_df['box-cox'].mean():.3f}")
print(f"Active: macro={MACRO_TRANSFORM!r},  micro={MICRO_TRANSFORM!r}")
"""),

        cell("## 5. Serialize Artifacts", "markdown"),

        cell("""\
joblib.dump(X_raw,        RESULTS_DIR / 'X_raw.pkl')
joblib.dump(food_ids,     RESULTS_DIR / 'food_ids.pkl')
joblib.dump(list(NUTRIENT_COLS), RESULTS_DIR / 'nutrient_cols.pkl')

print("Saved:")
for f in ['X_raw.pkl', 'food_ids.pkl', 'nutrient_cols.pkl']:
    path = RESULTS_DIR / f
    print(f"  {path}  ({path.stat().st_size} bytes)")
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# 03_masking.ipynb
# ══════════════════════════════════════════════════════════════════════════════
def make_03_masking():
    cells = [
        cell("# 03 — Masking\n"
             "Reproducibility anchor: generate three evaluation scenarios "
             "(Random 20%, Block-Micro, Cold-Start) as pure functions of "
             "`(X_raw, seed, params)`.", "markdown"),

        colab_setup_cell(),   # ← Colab setup

        cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import joblib

from config import (
    RESULTS_DIR,
    RANDOM_SEED, TEST_SEEDS,
    RANDOM_MASK_FRAC, BLOCK_MASK_FOOD_FRAC,
    COLD_START_COLS,
    MACRO_COLS, MICRO_COLS, NUTRIENT_COLS,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

X_raw         = joblib.load(RESULTS_DIR / 'X_raw.pkl')
nutrient_cols = joblib.load(RESULTS_DIR / 'nutrient_cols.pkl')
nutrient_list = list(nutrient_cols)
print(f"X_raw shape: {X_raw.shape}")
"""),

        cell("## Masking Functions", "markdown"),

        cell('''\
def create_random_mask(X: np.ndarray, frac: float, seed: int):
    """
    Randomly hold out `frac` fraction of *observed* (non-NaN) entries.
    Returns (mask_test, mask_train).
      mask_test  [bool] True  = held out for evaluation
      mask_train [bool] True  = available for model fitting
    """
    rng          = np.random.default_rng(seed)
    observed_idx = np.argwhere(~np.isnan(X))          # only real entries
    n_mask       = int(len(observed_idx) * frac)
    chosen       = rng.choice(len(observed_idx), size=n_mask, replace=False)

    mask_test = np.zeros(X.shape, dtype=bool)
    for r, c in observed_idx[chosen]:
        mask_test[r, c] = True

    mask_train = (~np.isnan(X)) & (~mask_test)

    # Integrity check
    assert not np.any(np.isnan(X[mask_test])), "Held-out entries must be observed!"
    return mask_test, mask_train


def create_block_mask(X: np.ndarray, food_frac: float,
                      micro_col_indices: list, seed: int):
    """
    Hold out ALL micronutrient values for `food_frac` of foods that have
    at least one observed micro value.
    """
    rng       = np.random.default_rng(seed)
    micro_X   = X[:, micro_col_indices]
    eligible  = np.where(~np.isnan(micro_X).all(axis=1))[0]
    n_foods   = int(len(eligible) * food_frac)
    chosen    = rng.choice(eligible, size=n_foods, replace=False)

    mask_test = np.zeros(X.shape, dtype=bool)
    for fi in chosen:
        for ci in micro_col_indices:
            if not np.isnan(X[fi, ci]):
                mask_test[fi, ci] = True

    mask_train = (~np.isnan(X)) & (~mask_test)
    assert not np.any(np.isnan(X[mask_test])), "Block mask: held-out must be observed!"
    return mask_test, mask_train


def create_cold_start_mask(X: np.ndarray, cold_cols: list, col_names: list):
    """
    Hold out ALL observed values in the specified columns (cold-start scenario).
    Deterministic — no seed needed.
    """
    col_indices = [col_names.index(c) for c in cold_cols]

    mask_test = np.zeros(X.shape, dtype=bool)
    for j in col_indices:
        mask_test[:, j] = ~np.isnan(X[:, j])

    mask_train = (~np.isnan(X)) & (~mask_test)
    assert not np.any(np.isnan(X[mask_test])), "Cold-start: held-out must be observed!"
    return mask_test, mask_train
'''),

        cell("## Column Index Mapping", "markdown"),

        cell("""\
macro_indices = [nutrient_list.index(c) for c in MACRO_COLS]
micro_indices = [nutrient_list.index(c) for c in MICRO_COLS]
print(f"Macro indices: {macro_indices}")
print(f"Micro indices: {micro_indices}")

# Cold-start: show missing rates to justify column selection
for col in COLD_START_COLS:
    j = nutrient_list.index(col)
    pct = np.isnan(X_raw[:, j]).mean() * 100
    obs = (~np.isnan(X_raw[:, j])).mean() * 100
    print(f"  {col}: observed={obs:.1f}%, missing={pct:.1f}%")
"""),

        cell("## Generate All Masks", "markdown"),

        cell("""\
masks = {}

# ── Scenario A: Random 20% ────────────────────────────────────────────────
for seed in [RANDOM_SEED] + TEST_SEEDS:
    mt, mtr = create_random_mask(X_raw, RANDOM_MASK_FRAC, seed)
    key = f'A_seed{seed}'
    masks[key] = {'test': mt, 'train': mtr,
                  'scenario': 'random', 'seed': seed}
    print(f"A seed={seed:4d}  held-out={mt.sum():5d}  "
          f"train-obs={mtr.sum():6d}")

# ── Scenario B: Block Micro ───────────────────────────────────────────────
print()
for seed in [RANDOM_SEED] + TEST_SEEDS:
    mt, mtr = create_block_mask(X_raw, BLOCK_MASK_FOOD_FRAC, micro_indices, seed)
    key = f'B_seed{seed}'
    masks[key] = {'test': mt, 'train': mtr,
                  'scenario': 'block', 'seed': seed}
    n_foods = mt.any(axis=1).sum()
    print(f"B seed={seed:4d}  held-out={mt.sum():5d}  "
          f"affected foods={n_foods:4d}")

# ── Scenario C: Cold-Start ───────────────────────────────────────────────
print()
mt, mtr = create_cold_start_mask(X_raw, COLD_START_COLS, nutrient_list)
masks['C_cold'] = {'test': mt, 'train': mtr,
                   'scenario': 'cold_start', 'cols': COLD_START_COLS}
for j, col in enumerate(COLD_START_COLS):
    ci  = nutrient_list.index(col)
    cnt = mt[:, ci].sum()
    print(f"C col={col:<12}  held-out={cnt:5d}")
"""),

        cell("## Verification", "markdown"),

        cell("""\
print("=== VERIFICATION: no NaN in held-out positions ===")
all_ok = True
for key, m in masks.items():
    vals = X_raw[m['test']]
    ok   = not np.any(np.isnan(vals))
    status = "✓" if ok else "✗ FAILED"
    print(f"  {key:<18} held-out={m['test'].sum():5d}  {status}")
    if not ok:
        all_ok = False
print()
print("All checks passed:", all_ok)
"""),

        cell("## Save", "markdown"),

        cell("""\
joblib.dump(masks,        RESULTS_DIR / 'masks.pkl')
joblib.dump(COLD_START_COLS, RESULTS_DIR / 'cold_start_cols.pkl')
print(f"Saved {len(masks)} masks → {RESULTS_DIR / 'masks.pkl'}")
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# 04_baselines.ipynb
# ══════════════════════════════════════════════════════════════════════════════
def make_04_baselines():
    cells = [
        cell("# 04 — Baseline Imputation Methods\n"
             "Mean · Median · KNN (k∈{3,5,10}) · MICE (BayesianRidge)\n\n"
             "Each method is evaluated on all masking scenarios and test seeds. "
             "The preprocessor is **fit inside each fold** to prevent leakage.",
             "markdown"),

        colab_setup_cell(),   # ← Colab setup

        cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    RESULTS_DIR, RANDOM_SEED, TEST_SEEDS, NUTRIENT_COLS, NUTRIENT_TRANSFORM_MAP,
)
from utils import NutrientPreprocessor, evaluate_imputation

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

X_raw         = joblib.load(RESULTS_DIR / 'X_raw.pkl')
masks         = joblib.load(RESULTS_DIR / 'masks.pkl')
nutrient_cols = list(joblib.load(RESULTS_DIR / 'nutrient_cols.pkl'))
print(f"X_raw: {X_raw.shape}  |  masks: {len(masks)}")
"""),

        cell("## Baseline Runner", "markdown"),

        cell('''\
def run_baseline(X_raw, mask_train, mask_test, method, seed=42, **kwargs):
    """
    1. Fit NutrientPreprocessor on (observed & not held-out) entries only.
    2. Apply imputer on normalized matrix with held-out cells set to NaN.
    3. Inverse-transform → original scale.
    """
    prep = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
    prep.fit(X_raw, mask_train)

    X_norm  = prep.transform(X_raw)
    X_input = X_norm.copy()
    X_input[mask_test] = np.nan           # expose the held-out cells

    if method == 'mean':
        imp = SimpleImputer(strategy='mean',   keep_empty_features=True)
    elif method == 'median':
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
    elif method == 'knn':
        imp = KNNImputer(n_neighbors=kwargs.get('k', 5), weights='distance',
                         keep_empty_features=True)
    elif method == 'mice':
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            min_value=0.0,
            max_iter=10,
            random_state=seed,
            keep_empty_features=True,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    X_imputed_norm = imp.fit_transform(X_input)
    X_imputed      = prep.inverse_transform(X_imputed_norm)
    return X_imputed
'''),

        cell("## Evaluation Loop", "markdown"),

        cell("""\
# Methods to evaluate
method_configs = [
    ('mean',   {}),
    ('median', {}),
    ('knn',    {'k': 3}),
    ('knn',    {'k': 5}),
    ('knn',    {'k': 10}),
    ('mice',   {}),
]

# Scenarios: tuning seed (42) + test seeds
scenario_keys = {
    'A': [f'A_seed{s}' for s in [RANDOM_SEED] + TEST_SEEDS],
    'B': [f'B_seed{s}' for s in [RANDOM_SEED] + TEST_SEEDS],
    'C': ['C_cold'],
}

baseline_results = {}
total   = sum(len(v) for v in scenario_keys.values()) * len(method_configs)
counter = 0

for sc_name, mask_keys in scenario_keys.items():
    for mk in mask_keys:
        m            = masks[mk]
        mask_train   = m['train']
        mask_test    = m['test']

        for method, kwargs in method_configs:
            label    = method if not kwargs else f"{method}_k{kwargs.get('k','')}"
            counter += 1
            print(f"[{counter:3d}/{total}] {mk:<18} {label:<10}", end=" ... ")

            try:
                X_imp   = run_baseline(X_raw, mask_train, mask_test,
                                       method, seed=RANDOM_SEED, **kwargs)
                metrics = evaluate_imputation(X_raw, X_imp, mask_test,
                                              nutrient_cols)
                baseline_results[(mk, label)] = {
                    'metrics':  metrics,
                    'scenario': m.get('scenario', sc_name),
                    'seed':     m.get('seed'),
                    'method':   label,
                }
                print(f"median NRMSE = {metrics['median_nrmse']:.4f}")
            except Exception as exc:
                print(f"ERROR: {exc}")

print(f"\\nDone: {len(baseline_results)} result(s) collected.")
"""),

        cell("## Quick Summary", "markdown"),

        cell("""\
rows = []
for (mk, method), res in baseline_results.items():
    rows.append({
        'mask_key':    mk,
        'method':      method,
        'scenario':    res['scenario'],
        'median_nrmse': res['metrics']['median_nrmse'],
        'validity_pct': res['metrics']['validity_rate'] * 100,
    })
df_sum = pd.DataFrame(rows)

pivot = (df_sum[df_sum['mask_key'].str.contains('seed(?!42)', regex=True)]
               .groupby(['scenario', 'method'])['median_nrmse']
               .agg(['mean', 'std'])
               .round(4))
print("Median NRMSE (test seeds, mean ± std):")
print(pivot.to_string())
"""),

        cell("## Save", "markdown"),

        cell("""\
joblib.dump(baseline_results, RESULTS_DIR / 'baseline_results.pkl')
print(f"Saved {len(baseline_results)} results → {RESULTS_DIR / 'baseline_results.pkl'}")
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# 05_matrix_factorization.ipynb
# ══════════════════════════════════════════════════════════════════════════════
def make_05_mf():
    cells = [
        cell("# 05 — Matrix Factorization Imputation\n"
             "**SoftImpute** (nuclear-norm minimization via fancyimpute) and "
             "**Masked NMF** (custom multiplicative-update loop).\n\n"
             "Rank is tuned on Scenario A seed-42; the optimal rank is the "
             "quantitative evidence for **H1**.",
             "markdown"),

        colab_setup_cell(),   # ← Colab setup (installs fancyimpute)

        cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    RESULTS_DIR, RANDOM_SEED, TEST_SEEDS,
    SOFT_IMPUTE_RANK_RANGE, NMF_RANK_RANGE,
    NUTRIENT_COLS, NUTRIENT_TRANSFORM_MAP,
)
from utils import NutrientPreprocessor, evaluate_imputation

X_raw         = joblib.load(RESULTS_DIR / 'X_raw.pkl')
masks         = joblib.load(RESULTS_DIR / 'masks.pkl')
nutrient_cols = list(joblib.load(RESULTS_DIR / 'nutrient_cols.pkl'))

# ── Try importing fancyimpute ──────────────────────────────────────────────
try:
    from fancyimpute import SoftImpute
    HAS_SOFTIMPUTE = True
    print("fancyimpute.SoftImpute loaded ✓")
except ImportError:
    HAS_SOFTIMPUTE = False
    print("WARNING: fancyimpute not found. SoftImpute results will be skipped.")
    print("  Install with:  pip install fancyimpute")
"""),

        cell("## SoftImpute", "markdown"),

        cell('''\
def run_softimpute(X_raw, mask_train, mask_test, max_rank):
    """Leakage-safe SoftImpute pipeline."""
    prep   = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
    prep.fit(X_raw, mask_train)

    X_norm  = prep.transform(X_raw)
    X_input = X_norm.copy()
    X_input[mask_test] = np.nan               # fancyimpute expects np.nan

    imputer     = SoftImpute(max_rank=max_rank, verbose=False)
    X_completed = imputer.fit_transform(X_input)

    return prep.inverse_transform(X_completed)
'''),

        cell("""\
# Rank sweep on tuning fold (Scenario A, seed 42)
m_tune     = masks['A_seed42']
mask_train = m_tune['train']
mask_test  = m_tune['test']

rank_nrmse_si = {}

if HAS_SOFTIMPUTE:
    print("SoftImpute rank sweep (Scenario A, seed 42):")
    for r in SOFT_IMPUTE_RANK_RANGE:
        X_imp      = run_softimpute(X_raw, mask_train, mask_test, r)
        metrics    = evaluate_imputation(X_raw, X_imp, mask_test, nutrient_cols)
        rank_nrmse_si[r] = metrics['median_nrmse']
        print(f"  rank={r:2d}  median_NRMSE={metrics['median_nrmse']:.4f}")
    optimal_rank_si = min(rank_nrmse_si, key=rank_nrmse_si.get)
    print(f"\\nOptimal SoftImpute rank: {optimal_rank_si}  "
          f"(NRMSE={rank_nrmse_si[optimal_rank_si]:.4f})")
else:
    optimal_rank_si = 5
    print(f"SoftImpute skipped. Fallback optimal_rank = {optimal_rank_si}")
"""),

        cell("## Masked NMF (custom multiplicative updates)", "markdown"),

        cell('''\
def masked_nmf(X_obs_matrix: np.ndarray,
               mask_obs: np.ndarray,
               rank: int,
               n_iter: int = 300,
               eps: float = 1e-10,
               seed: int = 42) -> np.ndarray:
    """
    NMF with masked multiplicative update rules.
    Only gradients from observed (mask_obs==True) entries are used.

    X_obs_matrix : (n, p) — NaN replaced with 0 before calling
    mask_obs     : (n, p) bool — True = observed
    Returns      : (n, p) completed matrix W @ H
    """
    rng  = np.random.default_rng(seed)
    n, p = X_obs_matrix.shape

    W = rng.uniform(0, 1, (n, rank)).astype(np.float64) + eps
    H = rng.uniform(0, 1, (rank, p)).astype(np.float64) + eps

    X = np.where(mask_obs, X_obs_matrix, 0.0).astype(np.float64)
    M = mask_obs.astype(np.float64)

    prev_loss = np.inf
    for it in range(n_iter):
        WH = W @ H

        # H update
        numer_H = W.T @ (M * X)
        denom_H = W.T @ (M * WH) + eps
        H      *= numer_H / denom_H
        H       = np.maximum(H, eps)

        # W update
        WH      = W @ H
        numer_W = (M * X) @ H.T
        denom_W = (M * WH) @ H.T + eps
        W      *= numer_W / denom_W
        W       = np.maximum(W, eps)

        # Convergence check
        if it % 50 == 49:
            WH   = W @ H
            loss = float(np.sum(M * (X - WH) ** 2))
            if abs(prev_loss - loss) / (prev_loss + eps) < 1e-6:
                break
            prev_loss = loss

    return W @ H


def run_masked_nmf(X_raw, mask_train, mask_test, rank):
    """Leakage-safe Masked NMF pipeline."""
    prep   = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
    prep.fit(X_raw, mask_train)

    X_norm  = prep.transform(X_raw)
    X_input = X_norm.copy()
    X_input[mask_test] = np.nan

    mask_obs   = ~np.isnan(X_input)
    X_for_nmf  = np.where(mask_obs, X_input, 0.0)
    X_for_nmf  = np.maximum(X_for_nmf, 0.0)      # NMF requires non-negative

    X_completed = masked_nmf(X_for_nmf, mask_obs, rank=rank, seed=RANDOM_SEED)
    return prep.inverse_transform(X_completed)
'''),

        cell("""\
# NMF rank sweep on tuning fold
rank_nrmse_nmf = {}
print("Masked NMF rank sweep (Scenario A, seed 42):")
for r in NMF_RANK_RANGE:
    try:
        X_imp   = run_masked_nmf(X_raw, mask_train, mask_test, rank=r)
        metrics = evaluate_imputation(X_raw, X_imp, mask_test, nutrient_cols)
        rank_nrmse_nmf[r] = metrics['median_nrmse']
        print(f"  rank={r:2d}  median_NRMSE={metrics['median_nrmse']:.4f}")
    except Exception as exc:
        rank_nrmse_nmf[r] = np.nan
        print(f"  rank={r:2d}  ERROR: {exc}")

valid_nmf      = {r: v for r, v in rank_nrmse_nmf.items() if not np.isnan(v)}
optimal_rank_nmf = min(valid_nmf, key=valid_nmf.get) if valid_nmf else 5
print(f"\\nOptimal NMF rank: {optimal_rank_nmf}  "
      f"(NRMSE={rank_nrmse_nmf.get(optimal_rank_nmf, float('nan')):.4f})")
"""),

        cell("""\
# Save tuning results (used in Fig 1 of notebook 06)
joblib.dump({
    'rank_nrmse_softimpute': rank_nrmse_si,
    'rank_nrmse_nmf':        rank_nrmse_nmf,
    'optimal_rank_softimpute': optimal_rank_si,
    'optimal_rank_nmf':        optimal_rank_nmf,
}, RESULTS_DIR / 'rank_tuning_results.pkl')
print("Saved rank_tuning_results.pkl")
"""),

        cell("## Full Experiment Loop (all scenarios, test seeds)", "markdown"),

        cell("""\
all_results = {}

scenario_keys = {
    'random':     [f'A_seed{s}' for s in TEST_SEEDS],
    'block':      [f'B_seed{s}' for s in TEST_SEEDS],
    'cold_start': ['C_cold'],
}

for scenario, mask_keys in scenario_keys.items():
    for mk in mask_keys:
        m  = masks[mk]
        mt = m['test']
        mtr = m['train']

        # ── SoftImpute ──────────────────────────────────────────────────
        if HAS_SOFTIMPUTE:
            try:
                X_imp   = run_softimpute(X_raw, mtr, mt, max_rank=optimal_rank_si)
                metrics = evaluate_imputation(X_raw, X_imp, mt, nutrient_cols)
                all_results[(mk, 'softimpute')] = {
                    'metrics':  metrics,
                    'scenario': scenario,
                    'seed':     m.get('seed'),
                    'method':   'softimpute',
                    'rank':     optimal_rank_si,
                }
                print(f"SoftImpute  {mk:<18} NRMSE={metrics['median_nrmse']:.4f}")
            except Exception as exc:
                print(f"SoftImpute  {mk}: ERROR {exc}")

        # ── Masked NMF ──────────────────────────────────────────────────
        try:
            X_imp   = run_masked_nmf(X_raw, mtr, mt, rank=optimal_rank_nmf)
            metrics = evaluate_imputation(X_raw, X_imp, mt, nutrient_cols)
            all_results[(mk, 'masked_nmf')] = {
                'metrics':  metrics,
                'scenario': scenario,
                'seed':     m.get('seed'),
                'method':   'masked_nmf',
                'rank':     optimal_rank_nmf,
            }
            print(f"Masked NMF  {mk:<18} NRMSE={metrics['median_nrmse']:.4f}")
        except Exception as exc:
            print(f"Masked NMF  {mk}: ERROR {exc}")

print(f"\\nTotal MF results: {len(all_results)}")
"""),

        cell("## Save All MF Results", "markdown"),

        cell("""\
joblib.dump(all_results, RESULTS_DIR / 'all_results.pkl')
print(f"Saved {len(all_results)} results → {RESULTS_DIR / 'all_results.pkl'}")
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# 06_evaluation_reporting.ipynb
# ══════════════════════════════════════════════════════════════════════════════
def make_06_reporting():
    cells = [
        cell("# 06 — Evaluation & Reporting\n"
             "Generates all 5 paper figures and 4 tables from saved results.",
             "markdown"),

        colab_setup_cell(),   # ← Colab setup

        cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from scipy import stats
from scipy.linalg import svd as scipy_svd
import warnings
warnings.filterwarnings('ignore')

from config import (
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    RANDOM_SEED, TEST_SEEDS, NUTRIENT_COLS,
    MACRO_COLS, MICRO_COLS, COLD_START_COLS,
)

for d in [FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 12,
})

# Load artefacts
X_raw          = joblib.load(RESULTS_DIR / 'X_raw.pkl')
masks          = joblib.load(RESULTS_DIR / 'masks.pkl')
nutrient_cols  = list(joblib.load(RESULTS_DIR / 'nutrient_cols.pkl'))
baseline_res   = joblib.load(RESULTS_DIR / 'baseline_results.pkl')
all_res        = joblib.load(RESULTS_DIR / 'all_results.pkl')
rank_tuning    = joblib.load(RESULTS_DIR / 'rank_tuning_results.pkl')
cold_cols      = joblib.load(RESULTS_DIR / 'cold_start_cols.pkl')

try:
    from fancyimpute import SoftImpute
    BEST_MF = 'softimpute'
except ImportError:
    BEST_MF = 'masked_nmf'

print(f"Baseline results : {len(baseline_res)}")
print(f"MF results       : {len(all_res)}")
print(f"Best MF method   : {BEST_MF}")
"""),

        cell("# ── Fig 1: Singular Value Decay + Rank Tuning Curve (H1) ──",
             "markdown"),

        cell("""\
# Recompute SVD for Fig 1a
complete_rows = ~np.isnan(X_raw).any(axis=1)
X_c = X_raw[complete_rows]
X_z = (X_c - X_c.mean(0)) / (X_c.std(0) + 1e-10)
_, s, _ = scipy_svd(X_z, full_matrices=False)
cum_var = np.cumsum(s**2) / np.sum(s**2) * 100
k90 = int(np.argmax(cum_var >= 90)) + 1

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel a: scree + cumulative ──────────────────────────────────────────
ax, ax_r = axes[0], axes[0].twinx()
rnks = np.arange(1, len(s) + 1)
ax.bar(rnks, s**2 / s[0]**2, color='steelblue', alpha=0.6,
       label='Norm. eigenvalue')
ax_r.plot(rnks, cum_var, 'o-', color='tomato', ms=4,
          label='Cumul. variance')
ax_r.axhline(90, color='grey', ls='--', lw=0.8)
ax_r.axhline(80, color='grey', ls=':', lw=0.8)
ax.set_xlabel('Rank')
ax.set_ylabel('Normalised Eigenvalue', color='steelblue')
ax_r.set_ylabel('Cumulative Variance (%)', color='tomato')
lines  = ax.get_legend_handles_labels()
lines2 = ax_r.get_legend_handles_labels()
ax.legend(lines[0]+lines2[0], lines[1]+lines2[1], fontsize=8, loc='upper right')

# ── Panel b: rank tuning curve ───────────────────────────────────────────
ax2 = axes[1]
si_nrmse  = rank_tuning.get('rank_nrmse_softimpute', {})
nmf_nrmse = rank_tuning.get('rank_nrmse_nmf', {})

if si_nrmse and any(v is not None for v in si_nrmse.values()):
    valid   = {r: v for r, v in si_nrmse.items() if v is not None}
    best_si = min(valid, key=valid.get)
    ax2.plot(list(valid), list(valid.values()), 'o-', color='steelblue',
             ms=5, label='SoftImpute')
    ax2.axvline(best_si, color='steelblue', ls='--', alpha=0.6,
                label=f'opt r={best_si}')

valid_nmf = {r: v for r, v in nmf_nrmse.items()
             if v is not None and not np.isnan(v)}
if valid_nmf:
    best_nmf = min(valid_nmf, key=valid_nmf.get)
    ax2.plot(list(valid_nmf), list(valid_nmf.values()), 's-', color='tomato',
             ms=5, label='Masked NMF')
    ax2.axvline(best_nmf, color='tomato', ls='--', alpha=0.6,
                label=f'opt r={best_nmf}')

ax2.set_xlabel('Rank')
ax2.set_ylabel('Median NRMSE')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_svd_rank_tuning.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Fig 1 saved  |  H1: top-{k90} rank explains 90% variance")
"""),

        cell("# ── Fig 2: NRMSE Box Plots per Method (H2) ──", "markdown"),

        cell("""\
fig2_rows = []

# Baselines — test seeds only (exclude seed-42 tuning fold)
for (mk, method), res in baseline_res.items():
    if res['scenario'] == 'random' and 'seed42' not in mk:
        fig2_rows.append({'method': method,
                          'median_nrmse': res['metrics']['median_nrmse'],
                          'type': 'baseline'})

# MF methods — all test seeds
for (mk, method), res in all_res.items():
    if res['scenario'] == 'random':
        fig2_rows.append({'method': method,
                          'median_nrmse': res['metrics']['median_nrmse'],
                          'type': 'mf'})

fig2_df = pd.DataFrame(fig2_rows)
method_order = ['mean', 'median', 'knn_k3', 'knn_k5', 'knn_k10',
                'mice', 'masked_nmf', 'softimpute']
avail = [m for m in method_order if m in fig2_df['method'].values]

palette = {m: ('#2196F3' if m in ('softimpute', 'masked_nmf')
               else '#EF9A9A')
           for m in avail}

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=fig2_df, x='method', y='median_nrmse',
            order=avail, palette=palette, ax=ax,
            flierprops={'marker': 'o', 'markersize': 4})
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_xlabel('Method')
ax.set_ylabel('Median NRMSE')
ax.grid(axis='y', alpha=0.3)
from matplotlib.patches import Patch
handles = [Patch(color='#2196F3', label='Matrix Factorization'),
           Patch(color='#EF9A9A', label='Baseline')]
ax.legend(handles=handles, fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_nrmse_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig 2 saved")
"""),

        cell("# ── Fig 3: Observed Fraction vs NRMSE per Nutrient (H3) ──",
             "markdown"),

        cell("""\
obs_fracs = {col: (~np.isnan(X_raw[:, j])).mean()
             for j, col in enumerate(nutrient_cols)}

pn_rows = []
for (mk, method), res in all_res.items():
    if method == BEST_MF and res['scenario'] == 'random':
        for col, pn in res['metrics']['per_nutrient'].items():
            pn_rows.append({'nutrient': col, 'nrmse': pn['nrmse'],
                            'obs_frac': obs_fracs.get(col, np.nan)})

if pn_rows:
    pn_df  = pd.DataFrame(pn_rows)
    pn_agg = pn_df.groupby('nutrient').agg(
        nrmse=('nrmse', 'mean'), obs_frac=('obs_frac', 'first')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pn_agg['obs_frac'], pn_agg['nrmse'],
               s=70, alpha=0.85, color='steelblue', zorder=3)

    for _, row in pn_agg.iterrows():
        ax.annotate(row['nutrient'],
                    (row['obs_frac'], row['nrmse']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8, color='#333')

    valid = pn_agg.dropna()
    if len(valid) >= 3:
        z   = np.polyfit(valid['obs_frac'], valid['nrmse'], 1)
        xln = np.linspace(valid['obs_frac'].min(), valid['obs_frac'].max(), 100)
        ax.plot(xln, np.poly1d(z)(xln), 'r--', alpha=0.6, label='Linear trend')
        r, pv = stats.pearsonr(valid['obs_frac'], valid['nrmse'])
        ax.text(0.05, 0.95, f'r = {r:.3f},  p = {pv:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(fontsize=9)

    ax.set_xlabel('Observed Fraction (before masking)')
    ax.set_ylabel(f'Mean NRMSE ({BEST_MF}, 4 seeds)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_obs_frac_vs_nrmse.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Fig 3 saved")
else:
    print("No per-nutrient data available for Fig 3 — run notebook 05 first.")
"""),

        cell("# ── Fig 4: KS-Statistic Heatmap (H2) ──", "markdown"),

        cell("""\
ks_accum = {}   # method → {nutrient: [ks_stat, ...]}

for (mk, method), res in list(baseline_res.items()) + list(all_res.items()):
    scenario = res.get('scenario', '')
    seed     = res.get('seed')
    # Only random-mask test seeds
    if scenario != 'random' or seed == RANDOM_SEED:
        continue
    ks_accum.setdefault(method, {col: [] for col in nutrient_cols})
    for col, pn in res['metrics']['per_nutrient'].items():
        ks_accum[method][col].append(pn['ks_stat'])

ks_rows = {}
for method, col_data in ks_accum.items():
    ks_rows[method] = {col: (np.mean(v) if v else np.nan)
                       for col, v in col_data.items()}

ks_df = pd.DataFrame(ks_rows).T[nutrient_cols]

if not ks_df.empty:
    h = max(3, len(ks_df) * 0.55 + 1)
    fig, ax = plt.subplots(figsize=(15, h))
    sns.heatmap(ks_df.round(3), annot=True, fmt='.2f',
                cmap='YlOrRd', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'KS Statistic (lower = better)'})
    ax.set_xlabel('Nutrient')
    ax.set_ylabel('Method')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_ks_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Fig 4 saved")
else:
    print("No test-seed data for Fig 4.")
"""),

        cell("# ── Fig 5: Cold-Start NRMSE vs Observed Fraction (H3) ──",
             "markdown"),

        cell("""\
cold_pn = []
for (mk, method), res in list(all_res.items()) + list(baseline_res.items()):
    if res.get('scenario') == 'cold_start':
        for col, pn in res['metrics']['per_nutrient'].items():
            cold_pn.append({'method': method, 'nutrient': col,
                            'nrmse': pn['nrmse'],
                            'obs_frac': obs_fracs.get(col, np.nan)})

if cold_pn:
    cold_df = pd.DataFrame(cold_pn)
    cold_cs = cold_df[cold_df['nutrient'].isin(cold_cols)]

    fig, ax = plt.subplots(figsize=(10, 6))
    palette_cs = sns.color_palette('tab10', n_colors=cold_cs['method'].nunique())
    for (method, grp), clr in zip(cold_cs.groupby('method'), palette_cs):
        ax.scatter(grp['obs_frac'], grp['nrmse'],
                   label=method, s=80, alpha=0.85, color=clr, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(row['nutrient'],
                        (row['obs_frac'], row['nrmse']),
                        textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax.set_xlabel('Observed Fraction (before cold-start masking)')
    ax.set_ylabel('NRMSE')
    ax.legend(fontsize=9, bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_coldstart_nrmse.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Fig 5 saved")
else:
    print("No cold-start data for Fig 5.")
"""),

        cell("# ── Tables ──", "markdown"),

        cell("""\
# Table 1 — Nutrient Summary
t1_path = TABLES_DIR / 'nutrient_summary.csv'
if t1_path.exists():
    t1 = pd.read_csv(t1_path, index_col=0)
    t1.to_csv(TABLES_DIR / 'table1_nutrient_summary.csv')
    print("Table 1 (Nutrient Summary):")
    print(t1.round(3).to_string())
else:
    print("nutrient_summary.csv not found — run 01_eda.ipynb first.")
"""),

        cell("""\
# Table 2 — Method comparison (random mask, test seeds)
rows_t2 = []
for (mk, method), res in baseline_res.items():
    if res['scenario'] == 'random' and 'seed42' not in mk:
        rows_t2.append({'method': method,
                        'nrmse': res['metrics']['median_nrmse'],
                        'validity': res['metrics']['validity_rate']})
for (mk, method), res in all_res.items():
    if res['scenario'] == 'random':
        rows_t2.append({'method': method,
                        'nrmse': res['metrics']['median_nrmse'],
                        'validity': res['metrics']['validity_rate']})

t2_df = pd.DataFrame(rows_t2)
if not t2_df.empty:
    t2 = t2_df.groupby('method').agg(
        nrmse_mean =('nrmse',    'mean'),
        nrmse_std  =('nrmse',    'std'),
        validity   =('validity', 'mean'),
    ).sort_values('nrmse_mean')
    t2['NRMSE (mean±std)'] = t2.apply(
        lambda r: f"{r['nrmse_mean']:.4f} ± {r['nrmse_std']:.4f}"
                  if not np.isnan(r['nrmse_std']) else f"{r['nrmse_mean']:.4f}", axis=1)
    t2['Validity %'] = (t2['validity'] * 100).round(1)
    print("Table 2 — Method Comparison (Random Mask):")
    print(t2[['NRMSE (mean±std)', 'Validity %']].to_string())
    t2.to_csv(TABLES_DIR / 'table2_method_comparison.csv')
"""),

        cell("""\
# Table 3 — Scenario comparison for best MF method
rows_t3 = []
for (mk, method), res in all_res.items():
    if method == BEST_MF:
        rows_t3.append({'scenario': res['scenario'],
                        'nrmse': res['metrics']['median_nrmse'],
                        'validity': res['metrics']['validity_rate']})

if rows_t3:
    t3 = (pd.DataFrame(rows_t3)
            .groupby('scenario')
            .agg(nrmse_mean=('nrmse','mean'),
                 nrmse_std =('nrmse','std'),
                 validity  =('validity','mean'))
            .round(4))
    print(f"Table 3 — Scenario Comparison ({BEST_MF}):")
    print(t3.to_string())
    t3.to_csv(TABLES_DIR / 'table3_scenario_comparison.csv')
"""),

        cell("""\
# Table 4 — Per-nutrient NRMSE: best MF vs best baseline (random mask, test seeds)
pn4 = []
for (mk, method), res in all_res.items():
    if method == BEST_MF and res['scenario'] == 'random':
        for col, pn in res['metrics']['per_nutrient'].items():
            pn4.append({'method': method, 'nutrient': col, 'nrmse': pn['nrmse']})
for (mk, method), res in baseline_res.items():
    if res['scenario'] == 'random' and 'seed42' not in mk:
        for col, pn in res['metrics']['per_nutrient'].items():
            pn4.append({'method': method, 'nutrient': col, 'nrmse': pn['nrmse']})

if pn4:
    pn4_df  = pd.DataFrame(pn4)
    pn4_avg = pn4_df.groupby(['method','nutrient'])['nrmse'].mean().reset_index()

    mf_pn  = pn4_avg[pn4_avg['method']==BEST_MF].set_index('nutrient')['nrmse']
    bl_pn  = (pn4_avg[pn4_avg['method']!=BEST_MF]
                      .groupby('nutrient')['nrmse'].min())
    t4 = pd.DataFrame({f'{BEST_MF}_nrmse': mf_pn,
                        'best_baseline_nrmse': bl_pn})
    t4['delta_pct'] = ((t4['best_baseline_nrmse'] - t4[f'{BEST_MF}_nrmse'])
                        / t4['best_baseline_nrmse'] * 100).round(1)
    print("Table 4 — Per-nutrient NRMSE (best MF vs best baseline):")
    print(t4.round(4).to_string())
    t4.to_csv(TABLES_DIR / 'table4_per_nutrient_comparison.csv')
"""),

        cell("# ── Wilcoxon Signed-Rank Test (H2 statistical support) ──",
             "markdown"),

        cell("""\
mf_scores = [res['metrics']['median_nrmse']
             for (mk, method), res in all_res.items()
             if method == BEST_MF and res['scenario'] == 'random']

bl_scores = []
for seed in TEST_SEEDS:
    mk  = f'A_seed{seed}'
    best_nrmse = min(
        (res['metrics']['median_nrmse']
         for (k, _), res in baseline_res.items()
         if k == mk),
        default=None
    )
    if best_nrmse is not None:
        bl_scores.append(best_nrmse)

print(f"MF  scores : {[round(v,4) for v in mf_scores]}")
print(f"BL  scores : {[round(v,4) for v in bl_scores]}")

n = min(len(mf_scores), len(bl_scores))
if n >= 2:
    stat, pval = stats.wilcoxon(mf_scores[:n], bl_scores[:n], alternative='less')
    print(f"\\nWilcoxon W = {stat:.4f},  p = {pval:.4f}")
    conclusion = "H2 SUPPORTED: MF significantly better" if pval < 0.05 \\
                 else "No significant difference at α=0.05"
    print(f"Conclusion : {conclusion}")
else:
    print("Not enough paired samples for Wilcoxon test.")
"""),

        cell("""\
print("\\n=== Pipeline complete ===")
print(f"Figures → {FIGURES_DIR}")
print(f"Tables  → {TABLES_DIR}")
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# Notebook 07 — Additional Analysis
# ══════════════════════════════════════════════════════════════════════════════
def make_07_additional_analysis():
    cells = [
        colab_setup_cell(),

        cell("# 07 — Additional Analysis\n"
             "Cumulative variance explained by SVD rank + transform ablation study.",
             "markdown"),

        # ── Imports & data load ───────────────────────────────────────────────
        cell("""\
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path('.').resolve()))

from config import (
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    RANDOM_SEED, TEST_SEEDS, NUTRIENT_COLS,
    MACRO_TRANSFORM, MICRO_TRANSFORM, NUTRIENT_TRANSFORM_MAP,
    SOFT_IMPUTE_RANK_RANGE,
)
from utils import NutrientPreprocessor, evaluate_imputation

for d in [FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

X_raw         = joblib.load(RESULTS_DIR / 'X_raw.pkl')
masks         = joblib.load(RESULTS_DIR / 'masks.pkl')
nutrient_cols = list(joblib.load(RESULTS_DIR / 'nutrient_cols.pkl'))
print(f"X_raw: {X_raw.shape}  |  masks: {len(masks)}")
"""),

        # ── Section 1: Cumulative Variance Explained ──────────────────────────
        cell("## 1. Cumulative Variance Explained by SVD Rank", "markdown"),

        cell("""\
# Use complete rows only (no imputation artefacts)
complete_mask = ~np.isnan(X_raw).any(axis=1)
X_complete    = X_raw[complete_mask]
print(f"Complete rows: {complete_mask.sum()} / {X_raw.shape[0]}")

# Normalize with current NUTRIENT_TRANSFORM_MAP (leakage-free: all rows are 'training')
obs_mask = ~np.isnan(X_complete)
prep     = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
prep.fit(X_complete, obs_mask)
X_norm   = prep.transform(X_complete)
X_norm   = np.nan_to_num(X_norm, nan=0.0)   # safety; should be zero residual NaNs

# SVD on mean-centered matrix
X_centered        = X_norm - X_norm.mean(axis=0)
U, s, Vt          = np.linalg.svd(X_centered, full_matrices=False)
var_explained     = (s ** 2) / (s ** 2).sum()
cum_var_explained = np.cumsum(var_explained)

print("\\nCumulative variance explained by rank k:")
for k in [1, 2, 3, 4, 5, 10, 15]:
    print(f"  rank={k:2d}: {cum_var_explained[k-1]*100:.1f}%")
print(f"\\nOptimal rank selected in Notebook 05: 3")
print(f"Variance explained by rank 3: {cum_var_explained[2]*100:.1f}%")
"""),

        cell("""\
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left — individual variance per component
axes[0].bar(range(1, len(s)+1), var_explained*100,
            color='steelblue', alpha=0.8)
axes[0].axvline(3, color='crimson', linestyle='--', linewidth=1.5, label='rank = 3')
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_xlim(0.5, len(s)+0.5)
axes[0].legend()

# Right — cumulative
ranks = np.arange(1, len(cum_var_explained)+1)
axes[1].plot(ranks, cum_var_explained*100, 'o-', color='steelblue',
             markersize=4, linewidth=1.5)
axes[1].axvline(3, color='crimson', linestyle='--', linewidth=1.5)
axes[1].axhline(cum_var_explained[2]*100, color='crimson', linestyle='--',
                linewidth=1.5,
                label=f'rank 3 = {cum_var_explained[2]*100:.1f}%')
axes[1].set_xlabel("Rank")
axes[1].set_ylabel("Cumulative Variance Explained (%)")
axes[1].set_xlim(0.5, len(s)+0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig6_cumvar_explained.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: fig6_cumvar_explained.png")
"""),

        # ── Section 2: Transform Ablation ─────────────────────────────────────
        cell("## 2. Transform Ablation Study", "markdown"),

        cell("""\
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer

try:
    from fancyimpute import SoftImpute
    HAS_SOFTIMPUTE = True
except ImportError:
    HAS_SOFTIMPUTE = False
    print("WARNING: fancyimpute not found — SoftImpute rows will be NaN")

# Three transform configurations to compare
TRANSFORM_CONFIGS = {
    'log1p (all)':              ['log1p']       * len(NUTRIENT_COLS),
    'yeo-johnson (all)':        ['yeo-johnson'] * len(NUTRIENT_COLS),
    'macro-YJ / micro-log1p':   NUTRIENT_TRANSFORM_MAP,
}

# Optimal ranks from Notebook 05 — update if your run produced different values
OPTIMAL_RANK_SI  = 3
OPTIMAL_RANK_NMF = 3
print(f"SoftImpute rank: {OPTIMAL_RANK_SI}  |  Masked NMF rank: {OPTIMAL_RANK_NMF}")
print(f"Configs: {list(TRANSFORM_CONFIGS.keys())}")
"""),

        # Masked NMF helper (self-contained copy)
        cell('''\
def masked_nmf(X_obs, mask_obs, rank, n_iter=300, eps=1e-10, seed=42):
    """Masked NMF with multiplicative updates (leakage-safe)."""
    rng  = np.random.default_rng(seed)
    n, p = X_obs.shape
    W    = rng.uniform(0, 1, (n, rank)).astype(np.float64) + eps
    H    = rng.uniform(0, 1, (rank, p)).astype(np.float64) + eps
    X    = np.where(mask_obs, X_obs, 0.0).astype(np.float64)
    M    = mask_obs.astype(np.float64)
    prev_loss = np.inf
    for it in range(n_iter):
        WH = W @ H
        H *= (W.T @ (M * X))  / (W.T @ (M * WH)  + eps)
        H  = np.maximum(H, eps)
        WH = W @ H
        W *= ((M * X) @ H.T)  / ((M * WH)  @ H.T + eps)
        W  = np.maximum(W, eps)
        if it % 50 == 49:
            loss = float(np.sum(M * (X - W @ H) ** 2))
            if abs(prev_loss - loss) / (prev_loss + eps) < 1e-6:
                break
            prev_loss = loss
    return W @ H
'''),

        cell('''\
def run_one_seed(X_raw, masks, seed, col_methods, rank_si, rank_nmf):
    """Run KNN-k5, SoftImpute, and Masked NMF for one seed with given col_methods."""
    m          = masks[f'A_seed{seed}']
    mask_train = m['train']
    mask_test  = m['test']

    prep    = NutrientPreprocessor(col_methods=col_methods)
    prep.fit(X_raw, mask_train)
    X_norm  = prep.transform(X_raw)
    X_input = X_norm.copy()
    X_input[mask_test] = np.nan

    out = {}

    # KNN-k5
    knn   = KNNImputer(n_neighbors=5, weights='distance', keep_empty_features=True)
    X_imp = prep.inverse_transform(knn.fit_transform(X_input))
    out['knn_k5'] = evaluate_imputation(X_raw, X_imp, mask_test, nutrient_cols)['median_nrmse']

    # SoftImpute
    if HAS_SOFTIMPUTE:
        X_si = SoftImpute(max_rank=rank_si, verbose=False).fit_transform(X_input)
        out['softimpute'] = evaluate_imputation(
            X_raw, prep.inverse_transform(X_si), mask_test, nutrient_cols)['median_nrmse']
    else:
        out['softimpute'] = np.nan

    # Masked NMF
    mask_obs  = ~np.isnan(X_input)
    X_for_nmf = np.maximum(np.where(mask_obs, X_input, 0.0), 0.0)
    X_nmf     = masked_nmf(X_for_nmf, mask_obs, rank=rank_nmf, seed=RANDOM_SEED)
    out['masked_nmf'] = evaluate_imputation(
        X_raw, prep.inverse_transform(X_nmf), mask_test, nutrient_cols)['median_nrmse']

    return out


print("Running ablation: 3 configs x 4 seeds x 3 methods ...")
ablation_raw = {
    cfg: {m: [] for m in ['knn_k5', 'softimpute', 'masked_nmf']}
    for cfg in TRANSFORM_CONFIGS
}

for cfg_name, col_methods in TRANSFORM_CONFIGS.items():
    for seed in TEST_SEEDS:
        res = run_one_seed(X_raw, masks, seed, col_methods,
                           OPTIMAL_RANK_SI, OPTIMAL_RANK_NMF)
        for method, val in res.items():
            ablation_raw[cfg_name][method].append(val)
    print(f"  {cfg_name}  done")

print("Ablation complete.")
'''),

        cell("""\
METHODS = ['knn_k5', 'softimpute', 'masked_nmf']
METHOD_LABELS = {'knn_k5': 'KNN-k5', 'softimpute': 'SoftImpute', 'masked_nmf': 'Masked NMF'}

rows = []
for cfg_name in TRANSFORM_CONFIGS:
    row = {'transform_config': cfg_name}
    for m in METHODS:
        vals = [v for v in ablation_raw[cfg_name][m] if not np.isnan(v)]
        if vals:
            row[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            row[f'{m}_std']  = round(float(np.std(vals)),  4)
            row[f'{m}_fmt']  = f"{np.mean(vals):.4f} \u00b1 {np.std(vals):.4f}"
        else:
            row[f'{m}_mean'] = np.nan
            row[f'{m}_std']  = np.nan
            row[f'{m}_fmt']  = 'N/A'
    rows.append(row)

abl_df = pd.DataFrame(rows)
abl_df.to_csv(TABLES_DIR / 'table5_transform_ablation.csv', index=False)

# Pretty-print
header = f"{'Transform Config':<28}" + "".join(f"{METHOD_LABELS[m]:>20}" for m in METHODS)
print("Transform Ablation — Median NRMSE (mean \u00b1 std, 4 seeds, Scenario A)")
print(header)
print("-" * (28 + 20*len(METHODS)))
for _, r in abl_df.iterrows():
    line = f"{r['transform_config']:<28}"
    line += "".join(f"{r[f'{m}_fmt']:>20}" for m in METHODS)
    print(line)
print("\\nSaved: table5_transform_ablation.csv")
"""),

        # ── Section 3: Missing Rate Sweep ──────────────────────────────────────
        cell("## 3. Missing Rate Sweep (5% – 50%)", "markdown"),

        cell("""\
def make_missing_rate_mask(X_raw, frac, seed):
    # Hold out `frac` of observed entries uniformly at random.
    # Only positions where X_raw is not NaN are eligible.
    rng          = np.random.default_rng(seed)
    observed_idx = np.argwhere(~np.isnan(X_raw))   # positions with real values
    n_hold       = int(len(observed_idx) * frac)
    chosen       = rng.choice(len(observed_idx), size=n_hold, replace=False)
    mask_test    = np.zeros(X_raw.shape, dtype=bool)
    for idx in observed_idx[chosen]:
        mask_test[idx[0], idx[1]] = True
    mask_train = (~np.isnan(X_raw)) & (~mask_test)
    return mask_train, mask_test


MISSING_FRACS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SWEEP_SEEDS    = TEST_SEEDS    # same 4 seeds as main experiment for consistency
SWEEP_METHODS  = ['knn_k5', 'softimpute', 'masked_nmf']
METHOD_COLORS  = {'knn_k5': 'steelblue', 'softimpute': 'crimson', 'masked_nmf': 'seagreen'}

print(f"Missing fracs : {[int(f*100) for f in MISSING_FRACS]}")
print(f"Seeds         : {SWEEP_SEEDS}  ({len(SWEEP_SEEDS)} seeds x {len(MISSING_FRACS)} fracs = {len(SWEEP_SEEDS)*len(MISSING_FRACS)} runs per method)")
print(f"Optimal ranks : SI={OPTIMAL_RANK_SI}  NMF={OPTIMAL_RANK_NMF}")
"""),

        cell('''\
print("Running missing-rate sweep ...")
# sweep_raw[frac][method] = list of NRMSE values over seeds
sweep_raw = {frac: {m: [] for m in SWEEP_METHODS} for frac in MISSING_FRACS}

for frac in MISSING_FRACS:
    for seed in SWEEP_SEEDS:
        mask_train, mask_test = make_missing_rate_mask(X_raw, frac, seed)

        prep    = NutrientPreprocessor(col_methods=NUTRIENT_TRANSFORM_MAP)
        prep.fit(X_raw, mask_train)
        X_norm  = prep.transform(X_raw)
        X_input = X_norm.copy()
        X_input[mask_test] = np.nan

        # KNN-k5
        knn   = KNNImputer(n_neighbors=5, weights='distance', keep_empty_features=True)
        X_imp = prep.inverse_transform(knn.fit_transform(X_input))
        sweep_raw[frac]['knn_k5'].append(
            evaluate_imputation(X_raw, X_imp, mask_test, nutrient_cols)['median_nrmse'])

        # SoftImpute
        if HAS_SOFTIMPUTE:
            X_si = SoftImpute(max_rank=OPTIMAL_RANK_SI, verbose=False).fit_transform(X_input)
            sweep_raw[frac]['softimpute'].append(
                evaluate_imputation(X_raw, prep.inverse_transform(X_si),
                                    mask_test, nutrient_cols)['median_nrmse'])
        else:
            sweep_raw[frac]['softimpute'].append(np.nan)

        # Masked NMF
        mask_obs  = ~np.isnan(X_input)
        X_for_nmf = np.maximum(np.where(mask_obs, X_input, 0.0), 0.0)
        X_nmf     = masked_nmf(X_for_nmf, mask_obs, rank=OPTIMAL_RANK_NMF, seed=seed)
        sweep_raw[frac]['masked_nmf'].append(
            evaluate_imputation(X_raw, prep.inverse_transform(X_nmf),
                                mask_test, nutrient_cols)['median_nrmse'])

    pct = int(frac * 100)
    means = {m: np.mean(sweep_raw[frac][m]) for m in SWEEP_METHODS}
    print(f"  {pct:3d}%  knn={means['knn_k5']:.4f}"
          f"  si={means['softimpute']:.4f}"
          f"  nmf={means['masked_nmf']:.4f}")

# Aggregate: mean and std per fraction
sweep_mean = {m: [np.mean(sweep_raw[f][m]) for f in MISSING_FRACS] for m in SWEEP_METHODS}
sweep_std  = {m: [np.std( sweep_raw[f][m]) for f in MISSING_FRACS] for m in SWEEP_METHODS}
print("Sweep complete.")
'''),

        cell("""\
pct_labels = [int(f*100) for f in MISSING_FRACS]

# Build tidy CSV: one row per (frac, seed, method)
rows = []
for frac in MISSING_FRACS:
    for i, seed in enumerate(SWEEP_SEEDS):
        for m in SWEEP_METHODS:
            rows.append({'missing_frac': frac, 'seed': seed,
                         'method': m, 'nrmse': sweep_raw[frac][m][i]})
sweep_df = pd.DataFrame(rows)
sweep_df.to_csv(TABLES_DIR / 'table6_missing_rate_sweep.csv', index=False)

# Summary table: mean ± std per (frac, method)
summary_rows = []
for frac, pct in zip(MISSING_FRACS, pct_labels):
    row = {'missing_pct': pct}
    for m in SWEEP_METHODS:
        row[f'{m}_mean'] = round(float(np.mean(sweep_raw[frac][m])), 4)
        row[f'{m}_std']  = round(float(np.std( sweep_raw[frac][m])), 4)
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Plot — mean line + shaded ±std band
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.array(pct_labels)

for m in SWEEP_METHODS:
    mu  = np.array(sweep_mean[m])
    sig = np.array(sweep_std[m])
    ax.plot(xs, mu, 'o-', color=METHOD_COLORS[m],
            label=METHOD_LABELS[m], linewidth=2, markersize=5)
    ax.fill_between(xs, mu - sig, mu + sig,
                    color=METHOD_COLORS[m], alpha=0.15)

ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel("Missing Rate (%)")
ax.set_ylabel("Median NRMSE (mean \u00b1 std, 4 seeds)")
ax.set_xticks(xs)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig7_missing_rate_sweep.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: fig7_missing_rate_sweep.png")
print("Saved: table6_missing_rate_sweep.csv")
"""),
    ]
    return notebook(cells)


# ══════════════════════════════════════════════════════════════════════════════
# Write all notebooks
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    specs = [
        ("01_eda.ipynb",                  make_01_eda()),
        ("02_preprocessing.ipynb",        make_02_preprocessing()),
        ("03_masking.ipynb",              make_03_masking()),
        ("04_baselines.ipynb",            make_04_baselines()),
        ("05_matrix_factorization.ipynb", make_05_mf()),
        ("06_evaluation_reporting.ipynb", make_06_reporting()),
        ("07_additional_analysis.ipynb",  make_07_additional_analysis()),
    ]

    for fname, nb in specs:
        out = ROOT / fname
        with open(out, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Written: {out}")

    print("\\nAll notebooks created successfully.")
