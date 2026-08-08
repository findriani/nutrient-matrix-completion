"""
make_figures.py — regenerate every figure from the finished CSVs.

Usage:
    python make_figures.py            # build all
    python make_figures.py sweep      # build one (sweep|mechanisms|downstream|
                                      #   coldstart|combined|boxplot|pica|runtime)

Reads outputs/tables/*.csv, writes outputs/figures/{name}.pdf + .png.
Style: figstyle.py (Okabe-Ito, large type, Tufte range-frames + direct labels).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs
from figstyle import st, OI, GREY, LGREY

TABLES = Path(__file__).resolve().parent.parent / "outputs" / "tables"
fs.apply_rc()


def _load(name):
    return pd.read_csv(TABLES / name)


def agg(df, val="nrmse", frac="missing_frac"):
    """mean / std / sem over seeds, per (method, frac)."""
    g = (df.groupby(["method", frac])[val]
           .agg(mean="mean", std="std", n="count").reset_index())
    g["sem"] = g["std"] / np.sqrt(g["n"])
    g["pct"] = g[frac] * 100
    return g


def _line(ax, g, method, band="sem", lw=2.6, ms=7, alpha_band=0.16, z=3):
    s = st(method)
    d = g[g.method == method].sort_values("pct")
    ax.plot(d.pct, d["mean"], color=s["c"], ls=s["ls"], marker=s["m"],
            markersize=ms, lw=lw, zorder=z, label=s["label"],
            markeredgecolor="white", markeredgewidth=0.6)
    if band and band in d:
        ax.fill_between(d.pct, d["mean"] - d[band], d["mean"] + d[band],
                        color=s["c"], alpha=alpha_band, lw=0, zorder=z - 1)
    return d


# ─────────────────────────────────────────────────────────────────────────────
def fig_sweep():
    """TKPI headline sweep: KNN, SoftImpute, Masked NMF, MissForest, PICA, 10 seeds."""
    main = _load("sweep_tkpi_10seed.csv")
    pica = _load("pica_sweep_10seed.csv")
    floor = _load("mean_median_sweep_tkpi.csv")
    wil = _load("wilcoxon_per_rate_tkpi.csv")
    g = agg(pd.concat([main, pica], ignore_index=True))
    gf = agg(floor)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    # tie zone
    ax.axvspan(15, 25, color="#F2F2F2", zorder=0)
    ax.text(20, 0.895, "transition\nband", ha="center", va="bottom",
            fontsize=12.5, color="#8A8A8A", style="italic")
    # mean floor
    mf = gf[gf.method == "mean"].sort_values("pct")
    ax.plot(mf.pct, mf["mean"], color=GREY, ls=(0, (1, 1.3)), lw=1.8, zorder=1)
    fs.direct_label(ax, 50, mf["mean"].iloc[-1] - 0.008, "mean", dx=0.7,
                    text="mean floor", fontsize=12.5)

    ends = {}
    for m in ["missforest", "masked_nmf", "pica", "knn_k10", "softimpute"]:
        d = _line(ax, g, m, lw=2.9 if m in ("knn_k10", "softimpute") else 2.0,
                  z=4 if m in ("knn_k10", "softimpute") else 3)
        ends[m] = d["mean"].iloc[-1]
    # de-collide the five end labels (knn/nmf cluster high, mf/si cluster mid, pica low)
    for m, yoff in [("knn_k10", 0.010), ("masked_nmf", -0.010),
                    ("missforest", 0.008), ("softimpute", -0.008),
                    ("pica", 0)]:
        fs.direct_label(ax, 50, ends[m] + yoff, m, dx=0.7)

    # significance asterisks (KNN vs SoftImpute, Holm-adjusted Wilcoxon per rate)
    ytop = 1.175
    sig_col = "sig_holm" if "sig_holm" in wil.columns else "sig"
    for _, r in wil.iterrows():
        if bool(r[sig_col]):
            ax.text(r["missing_frac"] * 100, ytop, "*", ha="center",
                    va="center", fontsize=17, color="#444444", fontweight="bold")
    ax.text(5, ytop + 0.012, "*  KNN-SoftImpute gap significant "
            "(Wilcoxon, Holm-adjusted p<0.05)", ha="left", va="bottom",
            fontsize=12, color="#444444")

    # transition band marker (interval language, not a point estimate) — scoped
    # explicitly to the KNN-SoftImpute pair, since PICA is also on this axis now
    ax.annotate("KNN-SoftImpute\ncrossover 20-25%", xy=(22.5, 0.982), xytext=(29, 0.925),
                fontsize=13, color="#1F1F1F", ha="left", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#7A7A7A", lw=1.4))

    ax.set_xlabel("Missing rate (%)")
    ax.set_ylabel("Median NRMSE  (mean ± SEM, 10 seeds)")
    ax.set_xlim(4, 58)
    ax.set_ylim(0.88, 1.19)
    ax.set_xticks(range(5, 51, 5))
    fs.tufte(ax)
    fs.range_frame(ax, xdata=[5, 50], ydata=[0.90, 1.15])
    fig.tight_layout()
    return fs.save(fig, "fig_sweep_tkpi")


# ─────────────────────────────────────────────────────────────────────────────
def fig_mechanisms():
    """4-panel: MCAR / MAR / MNAR (held-out) / MNAR (reference-norm) on USDA.

    Shared y-axis for MCAR + MAR; MNAR held-out on its own scale (inflated by
    denominator artefact); MNAR reference-norm on a moderate scale showing the
    transition reappears.
    """
    mcar = _load("sweep_usda_10seed.csv")
    mech = _load("sweep_usda_mechanisms_10seed.csv")
    methods = ["knn_k10", "softimpute"]

    # Prepare MNAR reference-norm as a separate "method-level" agg
    mnar_df = mech[mech.mechanism == "mnar"].copy()
    mnar_ref = mnar_df.copy()
    mnar_ref["nrmse"] = mnar_ref["nrmse_refnorm"]

    panels = [
        ("MCAR", mcar, "nrmse"),
        ("MAR", mech[mech.mechanism == "mar"], "nrmse"),
        ("MNAR (held-out norm)", mnar_df, "nrmse"),
        ("MNAR (reference norm)", mnar_ref, "nrmse"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    # share y-axis for first two panels (MCAR / MAR)
    axes[1].sharey(axes[0])

    import matplotlib.transforms as mtransforms
    # Transition interval per panel, matching the interval wording in the text.
    # None on the MNAR held-out panel, where the denominator artefact obscures
    # the transition (only the reference-normalised panel recovers it).
    bands = [(20, 25), (25, 30), None, (25, 30)]

    for ax, (name, df, val_col), band in zip(axes, panels, bands):
        g = agg(df, val=val_col)
        for m in methods:
            _line(ax, g, m, lw=2.7)
        ax.set_title(name, loc="center", pad=8)
        ax.set_xlabel("Missing rate (%)")
        ax.set_xlim(2, 52)
        ax.set_xticks(range(10, 51, 10))
        fs.tufte(ax)
        ax.axhline(1.0, color="#C9C9C9", lw=1.2, ls=(0, (4, 3)), zorder=1)
        # shade the transition interval rather than a single interpolated point
        if band is not None:
            lo, hi = band
            ax.axvspan(lo, hi, color="#F2F2F2", zorder=0)
            tr = mtransforms.blended_transform_factory(ax.transData,
                                                       ax.transAxes)
            ax.text((lo + hi) / 2, 0.90, f"{lo}-{hi}%", transform=tr,
                    ha="center", va="top", fontsize=12.5, color="#1F1F1F",
                    fontweight="bold")

    axes[0].set_ylabel("Median NRMSE  (mean +/- SEM)")
    # annotate MNAR held-out panel: "denominator artefact"
    axes[2].text(30, 3.3, "denominator\nartefact", ha="center", va="top",
                 fontsize=12, color="#8A8A8A", style="italic")
    # annotate MNAR ref-norm panel: "transition reappears"
    axes[3].text(35, 0.45, "transition\nreappears", ha="center", va="bottom",
                 fontsize=12, color="#8A8A8A", style="italic")
    axes[3].set_ylabel("Reference-norm NRMSE")

    # compact shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=st(m)["c"], lw=3, marker=st(m)["m"],
                      label=st(m)["label"]) for m in methods]
    axes[0].legend(handles=handles, loc="lower right", fontsize=13,
                   frameon=False)
    fig.tight_layout()
    return fs.save(fig, "fig_mechanisms_usda")


# ─────────────────────────────────────────────────────────────────────────────
def fig_downstream():
    """Slopegraph: method rank across 5 axes — point error, downstream acc/F1,
    covariance recovery, precision recovery."""
    d = _load("downstream_usda_summary.csv")
    cols = [("rank_nrmse", "Point\nerror"),
            ("rank_acc", "Downstream\naccuracy"),
            ("rank_f1", "Downstream\nmacro-F1"),
            ("rank_cov", "Covariance\nrecovery"),
            ("rank_prec", "Precision\nrecovery")]
    xs = list(range(len(cols)))

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for _, r in d.iterrows():
        m = r["method"]
        s = st(m)
        ys = [r[c] for c, _ in cols]
        emph = m in ("masked_nmf", "missforest", "knn_k10")
        ax.plot(xs, ys, color=s["c"], lw=3.2 if emph else 1.8,
                alpha=0.95 if emph else 0.55, zorder=4 if emph else 2,
                marker="o", markersize=8, markeredgecolor="white",
                markeredgewidth=0.8)
        # labels on left and right edges
        ax.text(-0.06, r[cols[0][0]], f"{s['label']}", ha="right",
                va="center", fontsize=12, color=s["c"],
                fontweight="bold" if emph else "normal")
        ax.text(xs[-1] + 0.06, r[cols[-1][0]], f"{s['label']}", ha="left",
                va="center", fontsize=12, color=s["c"],
                fontweight="bold" if emph else "normal")
    for x, (_, lab) in zip(xs, cols):
        ax.text(x, 0.25, lab, ha="center", va="bottom", fontsize=13.5,
                fontweight="bold", color="#222222")
    ax.set_xlim(-1.1, xs[-1] + 1.1)
    ax.set_ylim(9.8, 0.0)          # rank 1 at top; room for 9 methods
    ax.set_yticks(range(1, 10))
    ax.set_ylabel("Rank  (1 = best)")
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.tick_params(colors="#444444", length=4)
    fig.tight_layout()
    return fs.save(fig, "fig_downstream_usda")


# ─────────────────────────────────────────────────────────────────────────────
def fig_coldstart():
    """Cleveland dot plot: oracle vs donor vs naive per nutrient (log-x)."""
    d = _load("coldstart_scaling_regimes.csv").copy()
    d = d.sort_values("naive", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    reg = [("oracle", OI["green"], "Oracle (scaler sees held-out col)"),
           ("donor", OI["blue"], "Donor (USDA transfer)"),
           ("naive", OI["vermillion"], "Naive (no cold-start handling)")]

    fig, ax = plt.subplots(figsize=(9.8, 8.4))
    for yi, (_, row) in zip(y, d.iterrows()):
        ax.plot([row["oracle"], row["naive"]], [yi, yi], color="#DADADA",
                lw=1.4, zorder=1)
    for key, c, lab in reg:
        ax.scatter(d[key], y, s=70, color=c, label=lab, zorder=3,
                   edgecolor="white", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(d["nutrient"], fontsize=12)
    ax.set_xlabel("NRMSE on held-out nutrient (log scale)")
    ax.axvline(1.0, color="#BDBDBD", lw=1.1, ls=(0, (2, 2)), zorder=1)
    fs.tufte(ax, ygrid=False)
    ax.xaxis.grid(True, color="#ECECEC", lw=0.8)
    ax.legend(loc="lower right", fontsize=12.5, frameon=True,
              framealpha=0.95, edgecolor="#DDDDDD")
    fig.tight_layout()
    return fs.save(fig, "fig_coldstart_regimes")


# ─────────────────────────────────────────────────────────────────────────────
def fig_combined():
    """TKPI vs USDA side by side: the KNN→SoftImpute crossover generalises."""
    tk = _load("sweep_tkpi_10seed.csv")
    us = _load("sweep_usda_10seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)
    for ax, (name, df) in zip(axes, [("TKPI (1,146 × 19)", tk),
                                     ("USDA SR Legacy (7,793 × 18)", us)]):
        g = agg(df)
        for m in ["knn_k10", "softimpute"]:
            d = _line(ax, g, m, lw=2.9)
            fs.direct_label(ax, d.pct.iloc[-1], d["mean"].iloc[-1], m, dx=0.8)
        ax.set_title(name, loc="left", pad=8)
        ax.set_xlabel("Missing rate (%)")
        ax.set_xlim(4, 62)
        ax.set_xticks(range(10, 51, 10))
        fs.tufte(ax)
    axes[0].set_ylabel("Median NRMSE  (mean ± SEM)")
    fig.tight_layout()
    return fs.save(fig, "fig_crossover_combined")


# ─────────────────────────────────────────────────────────────────────────────
def fig_boxplot():
    """Scenario A (20%) NRMSE distribution per method, 10 seeds, sorted."""
    main = _load("main_scenarioA_10seed.csv").rename(
        columns={"median_nrmse": "nrmse"})[["seed", "method", "nrmse"]]
    base = _load("baselines_scenarioA_10seed.csv")[["seed", "method", "nrmse"]]
    pica = _load("pica_scenarioA_10seed.csv")[["seed", "method", "nrmse"]]
    d = pd.concat([main, base, pica], ignore_index=True)
    order = d.groupby("method").nrmse.median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for i, m in enumerate(order):
        vals = d[d.method == m].nrmse.values
        s = st(m)
        ax.scatter(np.full_like(vals, i, dtype=float) +
                   np.random.RandomState(0).uniform(-0.12, 0.12, len(vals)),
                   vals, color=s["c"], s=34, alpha=0.55, zorder=3,
                   edgecolor="white", linewidth=0.5)
        med = np.median(vals)
        ax.plot([i - 0.28, i + 0.28], [med, med], color=s["c"], lw=3.4, zorder=4)
    ax.axhline(1.0, color=GREY, ls=(0, (1, 1.3)), lw=1.6, zorder=1)
    ax.text(len(order) - 0.5, 1.003, "mean-imputation level", ha="right",
            va="bottom", fontsize=12, color=GREY, style="italic")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([st(m)["label"] for m in order], rotation=35, ha="right")
    ax.set_ylabel("Median NRMSE per seed")
    fs.tufte(ax)
    fig.tight_layout()
    return fs.save(fig, "fig_scenarioA_box")


# ─────────────────────────────────────────────────────────────────────────────
def fig_pica():
    """PICA collapses to the mean/median floor; structural methods beat it low."""
    main = _load("sweep_tkpi_10seed.csv")
    pica = _load("pica_sweep_10seed.csv")
    floor = _load("mean_median_sweep_tkpi.csv")
    g = agg(pd.concat([main, pica, floor], ignore_index=True))

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    # floor band = between mean and median
    gm = g[g.method == "mean"].sort_values("pct")
    gmd = g[g.method == "median"].sort_values("pct")
    ax.fill_between(gm.pct, gm["mean"].values, gmd["mean"].values,
                    color="#EDEDED", zorder=0)
    ax.text(7, 1.03, "mean–median floor", fontsize=12.5, color="#9A9A9A",
            style="italic", va="bottom")
    for m in ["knn_k10", "softimpute", "pica"]:
        d = _line(ax, g, m, lw=2.9 if m != "pica" else 2.6)
        fs.direct_label(ax, d.pct.iloc[-1], d["mean"].iloc[-1], m, dx=0.7)
    for m in ["mean", "median"]:
        d = g[g.method == m].sort_values("pct")
        ax.plot(d.pct, d["mean"], color=st(m)["c"], ls=st(m)["ls"], lw=1.6,
                zorder=1)
    ax.set_xlabel("Missing rate (%)")
    ax.set_ylabel("Median NRMSE  (mean ± SEM, 10 seeds)")
    ax.set_xlim(4, 58)
    ax.set_xticks(range(5, 51, 5))
    fs.tufte(ax)
    fig.tight_layout()
    return fs.save(fig, "fig_pica_floor")


# ─────────────────────────────────────────────────────────────────────────────
def fig_runtime():
    """Horizontal bar of wall-clock seconds per method (log-x)."""
    f = TABLES / "runtime.csv"
    if not f.exists():
        print("  runtime.csv not present yet — skipping fig_runtime")
        return None
    d = _load("runtime.csv").sort_values("sec_mean")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    ax.barh(y, d["sec_mean"], color=[st(m)["c"] for m in d.method],
            height=0.68, zorder=3, edgecolor="white")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["sec_mean"] * 1.08, yi, f"{r['sec_mean']:.2f}s", va="center",
                ha="left", fontsize=12, color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([st(m)["label"] for m in d.method])
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock seconds, TKPI @20% (mean of 3 seeds, log scale)")
    fs.tufte(ax, ygrid=False)
    ax.xaxis.grid(True, color="#ECECEC", lw=0.8)
    ax.set_xlim(right=d["sec_mean"].max() * 2.2)
    fig.tight_layout()
    return fs.save(fig, "fig_runtime")


ALL = {
    "sweep": fig_sweep, "mechanisms": fig_mechanisms, "downstream": fig_downstream,
    "coldstart": fig_coldstart, "combined": fig_combined, "boxplot": fig_boxplot,
    "pica": fig_pica, "runtime": fig_runtime,
}

if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(ALL)
    for name in which:
        fn = ALL.get(name)
        if fn is None:
            print(f"unknown figure '{name}' — options: {list(ALL)}")
            continue
        out = fn()
        if out:
            print(f"  [OK] {name:<11} -> {out.name}")
    print("done.")
