# Legacy pipeline

`create_notebooks.py` programmatically builds the exploratory-analysis notebooks
(EDA, preprocessing, masking, baseline imputers, SVD/rank diagnostics) that were
run before the experiments in [`../experiments/`](../experiments/) existed.
Executing it regenerates seven Jupyter notebooks from embedded cell
definitions (it uses `nbformat`, listed in the top-level `requirements.txt`).

Three figures in the paper trace back to this earlier stage rather than to
`experiments/`, because the analyses behind them were never rerun:

| Paper figure | Origin |
|---|---|
| Pipeline diagram (Fig. 2) | Manually authored (not code-generated); see `pipeline_diagram.txt` in the parent research directory for the text description it was drawn from. |
| Cumulative variance explained (Fig. 3) | `fig6_cumvar_explained.png`, produced by the SVD/PCA cell this script writes into the matrix-factorisation notebook. |
| SoftImpute rank-tuning curve (Fig. 4) | `fig1_svd_rank_tuning.png`, produced by the rank-sweep cell in the same notebook. |

All other reported numbers and figures come from `experiments/` and
supersede anything these notebooks computed earlier (baselines were rerun
at 10 seeds, more methods were added, and several protocol issues were
fixed there). Treat this folder as provenance for the three figures above,
not as a second source of results.

To regenerate the notebooks:

```bash
python create_notebooks.py
```

This writes `01_eda.ipynb` through `07_additional_analysis.ipynb` into the
working directory; run them in order with Jupyter. They expect the same raw
data files as the main pipeline (see the top-level README's "Data" section).
