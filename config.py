"""
config.py — Central configuration for TKPI Matrix Completion pipeline.
All seeds, paths, column lists, and hyperparameter grids are defined here.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
# Colab-safe: falls back to cwd() if __file__ is unavailable
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except (NameError, AttributeError):
    PROJECT_ROOT = Path.cwd()

DATA_PATH    = PROJECT_ROOT / "TKPIv2.csv"       # raw data at project root
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUT_DIR / "figures"
TABLES_DIR   = OUTPUT_DIR / "tables"
RESULTS_DIR  = OUTPUT_DIR / "results"

# ─── Column rename: Indonesian CSV headers → English ─────────────────────────
ID_COL = "KODE"   # food code column; kept as-is

COLUMN_RENAME = {
    "ENERGI":     "Energy",
    "PROTEIN":    "Protein",
    "LEMAK":      "Fat",
    "KH":         "Carbohydrate",
    "SERAT":      "Fiber",
    "KALSIUM":    "Calcium",
    "FOSFOR":     "Phosphorus",
    "BESI":       "Iron",
    "NATRIUM":    "Sodium",
    "KALIUM":     "Potassium",
    "TEMBAGA":    "Copper",
    "SENG":       "Zinc",
    "RETINOL":    "Retinol",
    "BKAR":       "Beta_Carotene",
    "KARTOTAL":   "Total_Carotenoids",
    "THIAMIN":    "Thiamine",
    "RIBOFLAVIN": "Riboflavin",
    "NIASIN":     "Niacin",
    "VIT_C":      "Vitamin_C",
}

# ─── Column definitions (English names, after rename) ────────────────────────
MACRO_COLS = [
    "Energy", "Protein", "Fat", "Carbohydrate", "Fiber",
]

MICRO_COLS = [
    "Calcium", "Phosphorus", "Iron", "Sodium", "Potassium",
    "Copper", "Zinc", "Retinol", "Beta_Carotene", "Total_Carotenoids",
    "Thiamine", "Riboflavin", "Niacin", "Vitamin_C",
]

NUTRIENT_COLS = MACRO_COLS + MICRO_COLS   # 19 nutrients total

# ─── Seeds & reproducibility ─────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SEEDS  = [123, 456, 789, 1024]

# ─── Masking parameters ──────────────────────────────────────────────────────
RANDOM_MASK_FRAC     = 0.20   # fraction of observed entries held out (Scenario A)
BLOCK_MASK_FOOD_FRAC = 0.30   # fraction of foods whose micro values are held out (B)

# Cold-start columns: 1 dense (Calcium ~96% observed),
#                     1 moderate (Thiamine ~92% observed → ~8% missing),
#                     1 sparse (Retinol ~47% observed → ~53% missing)
COLD_START_COLS = ["Calcium", "Thiamine", "Retinol"]

# ─── Hyperparameter grids ────────────────────────────────────────────────────
SOFT_IMPUTE_RANK_RANGE = range(1, 16)   # SoftImpute max_rank sweep
NMF_RANK_RANGE         = range(1, 16)   # Masked NMF rank sweep
KNN_K_VALUES           = [3, 5, 10]

# ─── Normalisation transform ─────────────────────────────────────────────────
# MinMaxScaler only — no power transform applied.
# Ablation experiments (notebook 07) showed power transforms provide negligible
# improvement for SoftImpute (the primary method) and add unnecessary complexity.
NUTRIENT_TRANSFORM_MAP = ['none'] * len(NUTRIENT_COLS)

# ─── Zero treatment ──────────────────────────────────────────────────────────
# True  → zeros are valid observed measurements (e.g., food truly has 0 Vitamin_C)
# False → zeros treated as missing (set to NaN before processing)
TREAT_ZEROS_AS_OBSERVED = True
