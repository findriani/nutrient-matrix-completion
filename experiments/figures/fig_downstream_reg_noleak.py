"""
fig_downstream_reg_noleak.py — Regenerate the downstream regression heatmap
using leakage-free results (target column excluded before imputation).

Reads: outputs/tables/downstream_{tkpi,usda}_regression_noleak_summary.csv
Writes: outputs/figures/fig_downstream_reg.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import figstyle as fs

TABLES = fs.FIG_DIR.parent / "tables"
fs.apply_rc()

# ── method display names (match the classification figure) ──────────────────
LABELS = {
    "mice": "MICE", "missforest": "MissForest", "pica": "PICA",
    "knn_k10": "KNN", "softimpute": "SoftImpute", "mean": "Mean",
    "median": "Median", "iterativesvd": "ISVD", "masked_nmf": "Masked NMF",
}

# ── load and sort by average R² ─────────────────────────────────────────────
tkpi = pd.read_csv(TABLES / "downstream_tkpi_regression_noleak_summary.csv")
usda = pd.read_csv(TABLES / "downstream_usda_regression_noleak_summary.csv")

# Sort by TKPI avg (descending) — same ordering for both panels
tkpi = tkpi.sort_values("avg", ascending=False).reset_index(drop=True)
method_order = tkpi["method"].tolist()
usda = usda.set_index("method").loc[method_order].reset_index()

# ── build matrices ──────────────────────────────────────────────────────────
cols = ["rf", "ridge", "knr"]
col_labels = ["RF", "Ridge", "KNR"]

tkpi_vals = tkpi[cols].values   # (n_methods, 3)
usda_vals = usda[cols].values
labels_y = [LABELS.get(m, m) for m in method_order]

# ── find column-best for bolding ────────────────────────────────────────────
tkpi_best = tkpi_vals.argmax(axis=0)  # best row per column
usda_best = usda_vals.argmax(axis=0)

# ── Okabe-Ito sequential colourmap ──────────────────────────────────────────
# Custom gradient: OI yellow (#F0E442) → OI skyblue (#56B4E9) → OI blue (#0072B2)
# Colour-blind safe and consistent with the paper's palette.
oi_cmap = LinearSegmentedColormap.from_list(
    "oi_seq",
    ["#F0E442", "#56B4E9", "#0072B2"],
    N=256,
)

all_vals = np.concatenate([tkpi_vals.ravel(), usda_vals.ravel()])
vmin, vmax = all_vals.min() - 0.02, all_vals.max() + 0.02
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ── plot ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8.5),
                                gridspec_kw={"width_ratios": [1, 1],
                                             "wspace": 0.12})

def draw_panel(ax, vals, best_rows, title, show_ylabels=True):
    n_rows, n_cols = vals.shape
    im = ax.imshow(vals, cmap=oi_cmap, norm=norm, aspect="auto")

    # annotate each cell
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            # text colour: dark on light cells, white on dark
            rgba = oi_cmap(norm(v))[:3]
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "black" if brightness > 0.55 else "white"
            weight = "bold" if i == best_rows[j] else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=14.5, color=tc, fontweight=weight)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=15)
    ax.set_yticks(range(n_rows))
    if show_ylabels:
        ax.set_yticklabels(labels_y, fontsize=15)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.tick_params(length=0)
    # remove all spines
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im

draw_panel(ax1, tkpi_vals, tkpi_best, "TKPI", show_ylabels=True)
im = draw_panel(ax2, usda_vals, usda_best, "USDA", show_ylabels=False)

# shared colourbar
cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.82, pad=0.025, aspect=28)
cbar.set_label("R²", fontsize=16, rotation=270, labelpad=18)
cbar.ax.tick_params(labelsize=13)

fig.tight_layout(rect=[0, 0, 0.92, 1])
fs.save(fig, "fig_downstream_reg")
print("Done — saved fig_downstream_reg.pdf + .png")
