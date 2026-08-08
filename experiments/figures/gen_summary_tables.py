"""
gen_summary_tables.py — Generate manuscript-ready summary tables from seed-level CSVs.

Produces 5 tables:
  1. tab_main_results.csv      — §5.2: method comparison at 20% MCAR (TKPI)
  2. tab_scenarios.csv         — §5.6: Scenario A vs B summary
  3. tab_sweep.csv             — §5.3: method × rate pivot (TKPI sweep, Table 11)
  4. tab_mechanisms.csv        — §5.4: mechanism × rate × method (USDA, both norms)
  5. tab_pica_usda.csv         — PICA on USDA MCAR rate-level summary
"""
import pandas as pd
import numpy as np
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT = T  # write summaries alongside the raw data


def fmt(mean, std):
    """Format as mean (±std) with consistent decimal places."""
    return f"{mean:.4f}"


# ── 1. tab_main_results: §5.2 method comparison at 20% ──────────────────────
def tab_main_results():
    main = pd.read_csv(T / "main_scenarioA_10seed.csv")
    main = main.rename(columns={"median_nrmse": "nrmse"})[["seed", "method", "nrmse"]]
    base = pd.read_csv(T / "baselines_scenarioA_10seed.csv")[["seed", "method", "nrmse"]]
    pica = pd.read_csv(T / "pica_scenarioA_10seed.csv")[["seed", "method", "nrmse"]]
    d = pd.concat([main, base, pica], ignore_index=True)
    g = d.groupby("method").nrmse.agg(["mean", "std", "median"]).reset_index()
    g = g.sort_values("mean")
    g["rank"] = range(1, len(g) + 1)
    g.columns = ["method", "mean_nrmse", "std_nrmse", "median_nrmse", "rank"]
    g.to_csv(OUT / "tab_main_results.csv", index=False, float_format="%.4f")
    print(f"  tab_main_results: {len(g)} methods")
    return g


# ── 2. tab_scenarios: §5.6 Scenario A vs B ──────────────────────────────────
def tab_scenarios():
    a = pd.read_csv(T / "main_scenarioA_10seed.csv")
    a = a.rename(columns={"median_nrmse": "nrmse"})
    b = pd.read_csv(T / "main_scenarioB_10seed.csv")
    b = b.rename(columns={"median_nrmse": "nrmse"})

    ga = a.groupby("method").nrmse.agg(["mean", "std"]).reset_index()
    ga.columns = ["method", "scenA_mean", "scenA_std"]
    gb = b.groupby("method").nrmse.agg(["mean", "std"]).reset_index()
    gb.columns = ["method", "scenB_mean", "scenB_std"]
    g = ga.merge(gb, on="method", how="outer").sort_values("scenA_mean")
    g.to_csv(OUT / "tab_scenarios.csv", index=False, float_format="%.4f")
    print(f"  tab_scenarios: {len(g)} methods")
    return g


# ── 3. tab_sweep: §5.3 Table 11 pivot ───────────────────────────────────────
def tab_sweep():
    main = pd.read_csv(T / "sweep_tkpi_10seed.csv")
    base = pd.read_csv(T / "baselines_sweep_10seed.csv")
    d = pd.concat([main, base], ignore_index=True)
    piv = d.pivot_table(index="missing_frac", columns="method",
                        values="nrmse", aggfunc="mean")
    # order columns by mean across all rates
    order = piv.mean().sort_values().index.tolist()
    piv = piv[order]
    piv.index = [f"{int(f*100)}%" for f in piv.index]
    piv.index.name = "missing_rate"
    piv.to_csv(OUT / "tab_sweep.csv", float_format="%.3f")
    print(f"  tab_sweep: {piv.shape[0]} rates x {piv.shape[1]} methods")
    return piv


# ── 4. tab_mechanisms: §5.4 mechanism summary ───────────────────────────────
def tab_mechanisms():
    d = pd.read_csv(T / "sweep_usda_mechanisms_10seed.csv")
    g = d.groupby(["mechanism", "missing_frac", "method"]).agg(
        nrmse_held=("nrmse", "mean"),
        nrmse_ref=("nrmse_refnorm", "mean"),
    ).reset_index()
    # pivot: rows = mechanism × rate, columns = method (held + ref)
    rows = []
    for mech in ["mcar", "mar", "mnar"]:
        for frac in sorted(g.missing_frac.unique()):
            sub = g[(g.mechanism == mech) & (g.missing_frac == frac)]
            row = {"mechanism": mech, "missing_rate": f"{int(frac*100)}%"}
            for _, r in sub.iterrows():
                row[f"{r.method}_held"] = r.nrmse_held
                row[f"{r.method}_ref"] = r.nrmse_ref
            # best method under held-out norm
            held_cols = {m: row.get(f"{m}_held", 99) for m in
                         ["knn_k5", "knn_k10", "softimpute"]}
            row["best_held"] = min(held_cols, key=held_cols.get)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tab_mechanisms.csv", index=False, float_format="%.3f")
    print(f"  tab_mechanisms: {len(out)} rows (3 mechs x 10 rates)")
    return out


# ── 5. tab_pica_usda: PICA USDA MCAR summary ────────────────────────────────
def tab_pica_usda():
    d = pd.read_csv(T / "pica_usda_mcar_10seed.csv")
    # also load KNN and SoftImpute from the USDA sweep for comparison
    usda = pd.read_csv(T / "sweep_usda_10seed.csv")
    usda = usda[usda.method.isin(["knn_k10", "softimpute"])]
    both = pd.concat([d, usda], ignore_index=True)
    piv = both.pivot_table(index="missing_frac", columns="method",
                           values="nrmse", aggfunc="mean")
    piv = piv[["knn_k10", "softimpute", "pica"]]
    piv.index = [f"{int(f*100)}%" for f in piv.index]
    piv.index.name = "missing_rate"
    piv.to_csv(OUT / "tab_pica_usda.csv", float_format="%.3f")
    print(f"  tab_pica_usda: {piv.shape[0]} rates x {piv.shape[1]} methods")
    return piv


if __name__ == "__main__":
    print("Generating manuscript-ready summary tables...\n")
    tab_main_results()
    tab_scenarios()
    tab_sweep()
    tab_mechanisms()
    tab_pica_usda()
    print("\nDone.")
