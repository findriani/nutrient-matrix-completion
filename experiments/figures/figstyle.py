"""
figstyle.py — shared plotting style for the figures below.

Design goals (per the paper's request):
  * Large, readable type (base 15 pt; nothing below 13 pt).
  * Okabe-Ito colour-blind-safe palette, ONE colour per method everywhere.
  * Tufte-inspired: maximal data-ink — no top/right spines, offset "range-frame"
    spines that span only the data, faint y-grid, direct line labelling instead
    of boxed legends where practical.

Every figure saves BOTH a vector PDF (for LaTeX includegraphics) and a 300-dpi
PNG (for quick preview) into outputs/figures/.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Okabe-Ito palette ────────────────────────────────────────────────────────
OI = dict(
    black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73",
    yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7",
)
GREY = "#8A8A8A"
LGREY = "#BDBDBD"

# ── One canonical style per method (colour / label / linestyle / marker) ──────
# The crossover pair uses the canonical high-contrast Okabe-Ito duo:
#   KNN = blue, SoftImpute = vermillion.
STYLE = {
    "knn_k10":      dict(c=OI["blue"],       label="KNN (k=10)",    ls="-",  m="o"),
    "knn_k5":       dict(c=OI["skyblue"],    label="KNN (k=5)",     ls="-",  m="s"),
    "knn_k3":       dict(c=OI["skyblue"],    label="KNN (k=3)",     ls="--", m="D"),
    "softimpute":   dict(c=OI["vermillion"], label="SoftImpute",    ls="-",  m="^"),
    "missforest":   dict(c=OI["green"],      label="MissForest",    ls="-",  m="D"),
    "masked_nmf":   dict(c=OI["purple"],     label="Masked NMF",    ls="-",  m="v"),
    "dae":          dict(c=OI["orange"],     label="DAE",           ls="-",  m="P"),
    "iterativesvd": dict(c=OI["black"],      label="IterativeSVD",  ls="-",  m="X"),
    "mice":         dict(c=OI["orange"],     label="MICE",          ls="-",  m="P"),
    "mice_extratrees": dict(c="#B07AA1",      label="MICE (ExtraTrees)", ls="-", m="P"),
    "pica":         dict(c=GREY,             label="PICA",          ls="-",  m="*"),
    "mean":         dict(c=GREY,             label="Mean",          ls=":",  m=None),
    "median":       dict(c=LGREY,            label="Median",        ls=":",  m=None),
}


def st(method):
    return STYLE.get(method, dict(c=GREY, label=method, ls="-", m="o"))


def apply_rc():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13.5,
        "legend.frameon": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def tufte(ax, ygrid=True):
    """Strip chartjunk; add a faint horizontal grid only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#444444", length=4)
    if ygrid:
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
        ax.xaxis.grid(False)


def range_frame(ax, xdata=None, ydata=None):
    """Offset spines that span only the data range (Tufte range frame)."""
    if xdata is not None:
        ax.spines["bottom"].set_bounds(min(xdata), max(xdata))
    if ydata is not None:
        ax.spines["left"].set_bounds(min(ydata), max(ydata))


def direct_label(ax, x, y, method, dx=0.6, va="center", fontsize=13.5, text=None):
    """Place a coloured method label at the end of a line (no legend box)."""
    s = st(method)
    ax.text(x + dx, y, text if text is not None else s["label"],
            color=s["c"], va=va, ha="left", fontsize=fontsize, fontweight="bold",
            clip_on=False)


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    return FIG_DIR / f"{name}.pdf"
