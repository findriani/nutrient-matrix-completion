"""
aggregate.py — Collate all experiment outputs into paper-ready summaries.

Reads every outputs/tables/*.csv produced by the scripts in this folder
and prints/writes: 10-seed sweep tables (TKPI + USDA) with the crossover point,
main Scenario A/B tables, per-nutrient comparison, Wilcoxon (n=10), MAR/MNAR
crossovers, downstream/covariance rankings, cold-start scaling regimes, and the
new-baseline comparison. Run after the experiment jobs complete.
"""
import numpy as np
import pandas as pd
from pathlib import Path

T = Path(__file__).resolve().parent / "outputs" / "tables"
OUT = Path(__file__).resolve().parent / "outputs" / "RESULTS_SUMMARY.md"
lines = []


def p(s=""):
    print(s)
    lines.append(str(s))


def sweep_table(csv, methods, title):
    if not (T / csv).exists():
        p(f"[missing] {csv}")
        return
    df = pd.read_csv(csv if Path(csv).is_absolute() else T / csv)
    p(f"\n### {title}")
    hdr = f"{'rate':>5} " + " ".join(f"{m:>12}" for m in methods) + "   crossover"
    p("```")
    p(hdr)
    prev_best = None
    cross = []
    for frac in sorted(df.missing_frac.unique()):
        sub = df[df.missing_frac == frac]
        means = {m: sub[sub.method == m].nrmse.mean() for m in methods}
        stds = {m: sub[sub.method == m].nrmse.std() for m in methods}
        best = min(means, key=means.get)
        row = f"{int(frac*100):>5} " + " ".join(
            f"{means[m]:.3f}±{stds[m]:.3f}" for m in methods)
        p(row + f"   BEST={best}")
        if prev_best and best != prev_best:
            cross.append((prev_best, best, int(frac*100)))
        prev_best = best
    p("```")
    for a, b, r in cross:
        p(f"- crossover {a} -> {b} at {r}% missing")
    return df


if __name__ == "__main__":
    p("# Experiment summary (10 seeds unless noted)\n")

    sweep_table("sweep_tkpi_10seed.csv",
                ['knn_k10', 'knn_k5', 'softimpute', 'masked_nmf', 'missforest'],
                "TKPI missing-rate sweep")
    sweep_table("sweep_usda_10seed.csv",
                ['knn_k10', 'knn_k5', 'softimpute'],
                "USDA MCAR sweep")

    # MAR/MNAR
    f = T / "sweep_usda_mechanisms_10seed.csv"
    if f.exists():
        dm = pd.read_csv(f)
        for mech in dm.mechanism.unique():
            sub = dm[dm.mechanism == mech].copy()
            sub.to_csv(T / f"_tmp_{mech}.csv", index=False)
            sweep_table(str(T / f"_tmp_{mech}.csv"),
                        ['knn_k10', 'knn_k5', 'softimpute'],
                        f"USDA {mech.upper()} sweep")
            (T / f"_tmp_{mech}.csv").unlink()

    # Main Scenario A / B
    for sc, fn in [("A (20% random)", "main_scenarioA_10seed.csv"),
                   ("B (block micro)", "main_scenarioB_10seed.csv")]:
        if (T / fn).exists():
            df = pd.read_csv(T / fn)
            g = df.groupby('method').agg(
                nrmse=('median_nrmse', 'mean'), nrmse_s=('median_nrmse', 'std'),
                mae=('mean_mae', 'mean'), mae_s=('mean_mae', 'std')).sort_values('nrmse')
            p(f"\n### Scenario {sc} main results")
            p("```")
            for m, r in g.iterrows():
                p(f"  {m:<16} NRMSE={r.nrmse:.4f}±{r.nrmse_s:.4f}   "
                  f"MAE={r.mae:.2f}±{r.mae_s:.2f}")
            p("```")

    # Baselines (iterativesvd, dae)
    if (T / "baselines_scenarioA_10seed.csv").exists():
        df = pd.read_csv(T / "baselines_scenarioA_10seed.csv")
        g = df.groupby('method').nrmse.agg(['mean', 'std'])
        p("\n### Additional baselines, Scenario A")
        p("```")
        for m, r in g.iterrows():
            p(f"  {m:<14} NRMSE={r['mean']:.4f}±{r['std']:.4f}")
        p("```")

    # Wilcoxon
    if (T / "wilcoxon_10seed.txt").exists():
        p("\n### Wilcoxon KNN k=10 vs SoftImpute, n=10")
        p("```")
        p((T / "wilcoxon_10seed.txt").read_text().strip())
        p("```")

    # Downstream
    if (T / "downstream_usda_summary.csv").exists():
        p("\n### Downstream classification + covariance, USDA")
        g = pd.read_csv(T / "downstream_usda_summary.csv", index_col=0)
        oracle = ""
        if (T / "downstream_oracle.txt").exists():
            oracle = (T / "downstream_oracle.txt").read_text().strip()
        p("```")
        p(oracle)
        p(g.round(4).to_string())
        p("```")

    # Cold-start
    if (T / "coldstart_scaling_regimes.csv").exists():
        df = pd.read_csv(T / "coldstart_scaling_regimes.csv")
        p("\n### Cold-start scaling regimes")
        p("```")
        for reg in ['oracle', 'donor', 'naive']:
            p(f"  {reg:<8} median NRMSE={df[reg].median():.4f}   "
              f"#<1.0={(df[reg] < 1.0).sum()}/19")
        p("```")
        p("oracle = held-out column scaled by its own observed range (the main-pipeline setting)")
        p("donor  = scaled by USDA range for same nutrient (external side info)")
        p("naive  = pooled range of other columns (no per-column scale prior)")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    p(f"\nWrote {OUT}")
