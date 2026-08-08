# Nuclear-Norm Matrix Completion for Nutrient Imputation

Reproducibility package for:

> **Characterising Nuclear-Norm Matrix Completion for Nutrient Imputation
> in Food Composition Databases**
>
> Indriani, Kartini, Budiman, Annisa, Mahmudah
> Universitas Lambung Mangkurat & Universitas Ahmad Dahlan

## Overview

National food composition databases record nutrient values per food item,
but a large share of entries, especially micronutrients, are never measured:
chemical analysis is expensive and lab capacity varies. This repository
contains the code to reproduce every experiment in the paper, which asks
three questions about how to fill in those gaps:

- **When does a global method beat a local one?** SoftImpute treats a
  missing nutrient value as recoverable from patterns shared across the
  whole database; KNN instead borrows values from the most similar foods.
  A masking sweep from 5% to 50% traces exactly where one overtakes the
  other.
- **Does it matter *why* a value is missing?** The same sweep is repeated
  under three different missingness mechanisms, completely random, tied to
  an observed food property, and tied to the missing value itself, to see
  whether the crossover point moves.
- **Does the most accurate method also work best in practice?** Every
  method is also tested on food-category classification and
  nutrient-prediction regression, since a database is rarely imputed just
  to admire the filled-in numbers.

Everything runs on two databases: TKPI, the Indonesian national food
composition table (1,146 foods, 19 nutrients), and USDA SR Legacy (7,793
foods, 18 nutrients).

## Requirements

```bash
pip install -r requirements.txt
```

CPU only throughout, including for the PyTorch autoencoder baseline; no GPU
is needed at this matrix scale.

## Data

Two databases, downloaded separately (not bundled here):

- **TKPI 2017** (1,146 foods x 19 nutrients): [huggingface.co/datasets/ULM-DS-Lab/food-composition-matrix](https://huggingface.co/datasets/ULM-DS-Lab/food-composition-matrix). Save as `TKPIv2.csv` in the repository root.
- **USDA SR Legacy** (7,793 foods x 18 nutrients, April 2018 release): [fdc.nal.usda.gov](https://fdc.nal.usda.gov/). Pivot to one row per food and one column per nutrient, `fdc_id` as the identifier, and save as `usda_sr_legacy_2018_wide.csv` in the repository root. Column names follow the list in `experiments/usda.py`.

Both files are expected next to `config.py`, not in a subfolder.

## Reproducing the experiments

Run scripts from inside `experiments/`; each inserts the repository root onto
its own import path. Output lands in `experiments/outputs/tables/`, and the
result CSVs already there let you check a fresh run against what's
reported without waiting for the slow methods to finish (see
"Reproducibility notes" below).

| Question | Script(s) | Key output |
|---|---|---|
| Which method reconstructs held-out values most accurately at a fixed 20% masking level? | `main_tkpi.py`, `baselines.py` | `main_scenarioA_10seed.csv` |
| At what masking level does SoftImpute (global) overtake KNN (local)? | `sweep_tkpi.py`, `pica_run.py`, `holm_hl.py`, `holm_hl_pica.py` | `sweep_tkpi_10seed.csv`, `wilcoxon_per_rate_tkpi.csv` |
| Does the same crossover show up on a database seven times larger? | `sweep_usda.py`, `pica_usda.py`, `holm_hl_pica_usda.py` | `sweep_usda_10seed.csv` |
| Does the crossover point move if values go missing for a different reason? | `sweep_mechanisms.py` | `sweep_usda_mechanisms_10seed.csv` |
| Is a nutrient that was never measured for *any* food still recoverable? | `coldstart.py` | `coldstart_scaling_regimes.csv` |
| Is masked NMF's instability a rank choice, or does it happen at every rank? | `nmf_rank_diag.py` | printed diagnostic |
| Does the most accurate method also win at food-category classification? | `downstream_tkpi_natural*.py`, `downstream_usda_natural*.py`, `downstream_mask_only.py` | `downstream_*_natural*_summary.csv` |
| ...or at predicting one nutrient from the rest? | `downstream_*regression*.py` | `downstream_*regression*_summary.csv` |
| How much does each method distort the relationships between nutrients? | `downstream.py`, `downstream_rates.py` | `downstream_usda_summary.csv` |
| How long does each method take to run? | `runtime.py` | `runtime.csv` |
| Collate the above into the paper's tables and figures | `aggregate.py`, `figures/gen_summary_tables.py`, `figures/make_figures.py` | `outputs/figures/*.png` |

A few smaller scripts (`mae.py`, `regression_avg_sd.py`, `repin.py`)
recompute specific supporting numbers; each explains itself in its own
docstring.

Two figures, the pipeline diagram and the cumulative-variance / rank-tuning
plots, come from an earlier notebook-based pipeline instead of the
experiments above; see [`legacy_pipeline/README.md`](legacy_pipeline/README.md).

## Repository structure

```
config.py, utils.py     preprocessing and scoring shared by every script below
experiments/             the experiments and figures in the paper
  common.py               data loading, masking, and the masked-NMF implementation
  pica.py                 PICA reimplementation (no official code was released)
  dl.py                   denoising-autoencoder baseline
  usda.py                 USDA loader + the three missingness mechanisms
  compat_patch.py         fixes for running fancyimpute on current scikit-learn
                           (see "Reproducibility notes")
  *.py                    one script per experiment, see the table above
  figures/                figure- and summary-table-generation scripts
  outputs/tables/         the result CSVs these scripts produced
legacy_pipeline/         earlier notebook-based pipeline (2 figures only)
```

## Reproducibility notes

Every method's own randomness, forest seeds, NMF initialisation, the
autoencoder's corruption mask, is fixed per run. The one exception needed
a patch rather than a seed argument: `fancyimpute`'s SoftImpute and
IterativeSVD call scikit-learn's randomized SVD with no fixed seed, so
`compat_patch.py` pins one. `repin.py` is a standalone check that the
patch is doing what it claims.

MissForest and MICE-ET are the slow methods, roughly 113 s and 144 s per
imputation call (`experiments/outputs/tables/runtime.csv`). `main_tkpi.py`,
`baselines.py`, and `sweep_tkpi.py` call these repeatedly and can take a
few hours on a laptop; everything else finishes in seconds.

## License

MIT (see [`LICENSE`](LICENSE)). TKPI and USDA SR Legacy are distributed
separately under their own respective terms; see the links above.
