"""
fig_downstream_clf_oi.py — Downstream classification heatmap (ACCURACY),
Okabe-Ito sequential colourmap (matching the regression figure).

As of the macro-F1 primary-metric switch, this accuracy heatmap is the
supplementary companion to fig_downstream_clf_f1.py, which is now the
figure included in the main manuscript. Kept unmodified (metric and output
name only) so the accuracy version can still be regenerated on demand.

Reads: outputs/tables/downstream_{tkpi,usda}_natural_{,lr_,nb_}summary.csv
Writes: outputs/figures/fig_downstream_clf_accuracy.{pdf,png}
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

# ── method display names ────────────────────────────────────────────────────
LABELS = {
    "mice": "MICE", "missforest": "MissForest", "pica": "PICA",
    "knn_k10": "KNN", "softimpute": "SoftImpute", "mean": "Mean",
    "median": "Median", "iterativesvd": "ISVD", "masked_nmf": "Masked NMF",
}

# ── load classification data ────────────────────────────────────────────────
# TKPI: RF from natural_summary, LR from natural_lr_summary, NB from natural_nb_summary
tkpi_rf = pd.read_csv(TABLES / "downstream_tkpi_natural_summary.csv")[["method", "clf_acc"]]
tkpi_lr = pd.read_csv(TABLES / "downstream_tkpi_natural_lr_summary.csv")[["method", "clf_acc"]]
tkpi_nb = pd.read_csv(TABLES / "downstream_tkpi_natural_nb_summary.csv")[["method", "clf_acc"]]

tkpi = tkpi_rf.rename(columns={"clf_acc": "rf"}).merge(
    tkpi_lr.rename(columns={"clf_acc": "lr"}), on="method").merge(
    tkpi_nb.rename(columns={"clf_acc": "nb"}), on="method")
tkpi["avg"] = tkpi[["rf", "lr", "nb"]].mean(axis=1)

# USDA: separate RF, LR, NB summary files
usda_rf = pd.read_csv(TABLES / "downstream_usda_natural_rf_summary.csv")[["method", "clf_acc"]]
usda_lr = pd.read_csv(TABLES / "downstream_usda_natural_lr_summary.csv")[["method", "clf_acc"]]
usda_nb = pd.read_csv(TABLES / "downstream_usda_natural_nb_summary.csv")[["method", "clf_acc"]]

usda = usda_rf.rename(columns={"clf_acc": "rf"}).merge(
    usda_lr.rename(columns={"clf_acc": "lr"}), on="method").merge(
    usda_nb.rename(columns={"clf_acc": "nb"}), on="method")
usda["avg"] = usda[["rf", "lr", "nb"]].mean(axis=1)

# ── sort order: match the manuscript table (Table downstream_clf) ───────────
method_order = ["mean", "median", "mice", "softimpute", "pica",
                "iterativesvd", "missforest", "knn_k10", "masked_nmf"]

tkpi = tkpi.set_index("method").loc[method_order].reset_index()
usda = usda.set_index("method").loc[method_order].reset_index()

# ── build matrices ──────────────────────────────────────────────────────────
cols = ["rf", "lr", "nb"]
col_labels = ["RF", "LR", "NB"]

tkpi_vals = tkpi[cols].values
usda_vals = usda[cols].values
labels_y = [LABELS.get(m, m) for m in method_order]

# ── find column-best for bolding ────────────────────────────────────────────
tkpi_best = tkpi_vals.argmax(axis=0)
usda_best = usda_vals.argmax(axis=0)

# ── Okabe-Ito sequential colourmap (same as regression figure) ──────────────
oi_cmap = LinearSegmentedColormap.from_list(
    "oi_seq", ["#F0E442", "#56B4E9", "#0072B2"], N=256)

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

    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
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
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im

draw_panel(ax1, tkpi_vals, tkpi_best, "TKPI (12 groups)", show_ylabels=True)
im = draw_panel(ax2, usda_vals, usda_best, "USDA (25 categories)", show_ylabels=False)

cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.82, pad=0.025, aspect=28)
cbar.set_label("Accuracy", fontsize=16, rotation=270, labelpad=18)
cbar.ax.tick_params(labelsize=13)

fig.tight_layout(rect=[0, 0, 0.92, 1])
fs.save(fig, "fig_downstream_clf_accuracy")
print("Done — saved fig_downstream_clf_accuracy.pdf + .png")
