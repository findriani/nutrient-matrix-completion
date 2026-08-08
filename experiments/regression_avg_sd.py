"""
regression_avg_sd.py -- mean +- SD (over 5 CV seeds) of the cross-target,
cross-regressor average R^2 for the downstream regression table, alongside
the mean already reported per method.

Post-hoc aggregation of already-collected per-seed rows in
downstream_{tkpi,usda}_regression_noleak.csv -- no new experiment.
For each (method, seed): average r2 across all target x regressor cells.
Then report mean +- SD of that per-seed average across the 5 seeds.
"""
import pandas as pd
from common import EXP_TABLES

for tag in ["tkpi", "usda"]:
    df = pd.read_csv(EXP_TABLES / f"downstream_{tag}_regression_noleak.csv")
    per_seed = df.groupby(["method", "cv_seed"])["r2"].mean().reset_index()
    agg = per_seed.groupby("method")["r2"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"=== {tag.upper()} (avg R^2 over 6 targets x 3 regressors, mean+-SD over 5 seeds) ===")
    for m, row in agg.iterrows():
        print(f"{m:14s}  {row['mean']:.3f} +- {row['std']:.3f}")
    print()
