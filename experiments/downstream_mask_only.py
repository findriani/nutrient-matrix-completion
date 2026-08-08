"""
downstream_mask_only.py — Mask-only classification diagnostic.

Tests whether the binary observation mask (1=observed, 0=missing) alone
predicts food category. If accuracy is high, mean/median winning
classification may reflect preserved missingness patterns rather than
nutrient recovery quality.

Protocol:
  M_ij = 1 if nutrient j is observed for food i, 0 otherwise.
  Train RF, LR, NB classifiers on M to predict food group/category.
  Compare with imputed-data classification accuracy.

Output: downstream_mask_only.csv (per-seed)
        downstream_mask_only_summary.csv
"""
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold

import sys; sys.path.insert(0, '.')
from common import load_tkpi, EXP_TABLES
from usda import load_usda
from _bootstrap import PROJECT_ROOT

CLF_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CLASSIFIERS = ['rf', 'lr', 'nb']


def make_clf(clf_type, seed):
    if clf_type == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=-1)
    elif clf_type == 'lr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, random_state=seed))
        ])
    elif clf_type == 'nb':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('nb', GaussianNB())
        ])


def clf_scores(X, y, clf_type, seed):
    clf = make_clf(clf_type, seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    r = cross_validate(clf, X, y, cv=cv, scoring='accuracy')
    return float(r['test_score'].mean())


if __name__ == "__main__":
    t0 = time.time()
    rows = []

    # === TKPI ===
    X_tkpi, ids_tkpi, cols_tkpi = load_tkpi()
    groups_tkpi = np.array([k[0] for k in ids_tkpi])
    # drop singleton Q
    keep = groups_tkpi != 'Q'
    X_tkpi = X_tkpi[keep]
    groups_tkpi = groups_tkpi[keep]

    # Binary mask: 1=observed, 0=missing
    M_tkpi = (~np.isnan(X_tkpi)).astype(float)

    n_classes = len(np.unique(groups_tkpi))
    miss_pct = 100 * np.isnan(X_tkpi).mean()
    print(f"TKPI: {X_tkpi.shape}, {n_classes} groups, {miss_pct:.1f}% missing")
    print(f"  Mask has {int(M_tkpi.sum(axis=0).min())}-{int(M_tkpi.sum(axis=0).max())} "
          f"observed per column (out of {M_tkpi.shape[0]})")

    # Check mask variance per column
    mask_var = M_tkpi.var(axis=0)
    n_informative = (mask_var > 0).sum()
    print(f"  {n_informative}/{M_tkpi.shape[1]} columns have mask variation")

    for clf_type in CLASSIFIERS:
        for seed in CLF_SEEDS:
            acc = clf_scores(M_tkpi, groups_tkpi, clf_type, seed)
            rows.append(dict(dataset='TKPI', classifier=clf_type, seed=seed,
                             accuracy=acc))
        accs = [r['accuracy'] for r in rows
                if r['dataset'] == 'TKPI' and r['classifier'] == clf_type]
        print(f"  Mask-only {clf_type:2s}: {np.mean(accs):.4f} "
              f"(±{np.std(accs):.4f})")

    # === USDA ===
    X_usda, ids_usda, cols_usda = load_usda()
    # Load food categories
    food = pd.read_csv(
        PROJECT_ROOT / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food.csv",
        usecols=['fdc_id', 'food_category_id'])
    cat = (pd.Series(ids_usda).to_frame('fdc_id')
           .merge(food, on='fdc_id', how='left').food_category_id.values)
    valid = ~pd.isna(cat)
    X_usda = X_usda[valid]
    cat_usda = cat[valid].astype(int)

    M_usda = (~np.isnan(X_usda)).astype(float)

    n_classes_u = len(np.unique(cat_usda))
    miss_pct_u = 100 * np.isnan(X_usda).mean()
    print(f"\nUSDA: {X_usda.shape}, {n_classes_u} categories, {miss_pct_u:.1f}% missing")
    print(f"  Mask has {int(M_usda.sum(axis=0).min())}-{int(M_usda.sum(axis=0).max())} "
          f"observed per column (out of {M_usda.shape[0]})")

    mask_var_u = M_usda.var(axis=0)
    n_informative_u = (mask_var_u > 0).sum()
    print(f"  {n_informative_u}/{M_usda.shape[1]} columns have mask variation")

    for clf_type in CLASSIFIERS:
        for seed in CLF_SEEDS:
            acc = clf_scores(M_usda, cat_usda, clf_type, seed)
            rows.append(dict(dataset='USDA', classifier=clf_type, seed=seed,
                             accuracy=acc))
        accs = [r['accuracy'] for r in rows
                if r['dataset'] == 'USDA' and r['classifier'] == clf_type]
        print(f"  Mask-only {clf_type:2s}: {np.mean(accs):.4f} "
              f"(±{np.std(accs):.4f})")

    # Save
    df = pd.DataFrame(rows)
    df.to_csv(EXP_TABLES / "downstream_mask_only.csv", index=False)

    summary = df.groupby(['dataset', 'classifier']).agg(
        acc_mean=('accuracy', 'mean'),
        acc_std=('accuracy', 'std')).reset_index()
    summary.to_csv(EXP_TABLES / "downstream_mask_only_summary.csv", index=False)

    # Print comparison table
    print("\n=== COMPARISON: Mask-only vs Imputed-data classification ===")
    print("(Imputed-data values from downstream_*_natural_*_summary.csv)\n")

    for ds in ['TKPI', 'USDA']:
        print(f"--- {ds} ---")
        print(f"{'Classifier':>10s}  {'Mask-only':>10s}  {'Best imputed':>12s}  "
              f"{'Worst imputed':>13s}")
        for ct in CLASSIFIERS:
            mask_acc = summary[(summary.dataset == ds) &
                               (summary.classifier == ct)].acc_mean.values[0]
            # Load imputed-data summaries for comparison
            prefix = 'tkpi' if ds == 'TKPI' else 'usda'
            try:
                imp_df = pd.read_csv(
                    EXP_TABLES / f"downstream_{prefix}_natural_{ct}_summary.csv")
                best = imp_df['accuracy_mean'].max()
                worst = imp_df['accuracy_mean'].min()
                print(f"{ct:>10s}  {mask_acc:>10.4f}  {best:>12.4f}  {worst:>13.4f}")
            except FileNotFoundError:
                # TKPI RF has different naming
                try:
                    imp_df = pd.read_csv(
                        EXP_TABLES / f"downstream_{prefix}_natural_summary.csv")
                    best = imp_df['accuracy_mean'].max()
                    worst = imp_df['accuracy_mean'].min()
                    print(f"{ct:>10s}  {mask_acc:>10.4f}  {best:>12.4f}  {worst:>13.4f}")
                except FileNotFoundError:
                    print(f"{ct:>10s}  {mask_acc:>10.4f}  (no comparison file)")
        print()

    print(f"\nDone in {time.time()-t0:.1f}s")
