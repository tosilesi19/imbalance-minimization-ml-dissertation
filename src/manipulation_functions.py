"""Experimental manipulation functions 

Implements the four factorial manipulations. Pipeline order:
imbalance -> minimization -> split -> metrics on test partition.
All functions are seed-deterministic. base_n = 15,000 for all
conditions so imbalance ratio is not confounded with sample size.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

RATIOS = {"50/50": 0.50, "80/20": 0.20, "90/10": 0.10, "95/5": 0.05}
LEVELS = {"100": 1.00, "75": 0.75, "50": 0.50, "25": 0.25}


def apply_imbalance(df, ratio_label, seed, base_n=15000):
    """Undersample to an exact class ratio at a fixed total size.

    ratio_label: one of '50/50', '80/20', '90/10', '95/5'.
    base_n: common total size across all conditions.
    """
    minority_frac = RATIOS[ratio_label]
    n_min = int(base_n * minority_frac)
    n_maj = base_n - n_min

    minority = df[df["target"] == 1].sample(n=n_min, random_state=seed)
    majority = df[df["target"] == 0].sample(n=n_maj, random_state=seed)

    out = pd.concat([minority, majority]).sample(frac=1, random_state=seed)
    return out.reset_index(drop=True)


def apply_horizontal_minimization(df, level_label, seed):
    """Retain a stratified fraction of rows.

    Stratification on target x sex_female preserves class and
    subgroup proportions under aggressive reduction.
    """
    frac = LEVELS[level_label]
    if frac == 1.00:
        return df.copy()

    strata = df["target"].astype(str) + "_" + df["sex_female"].astype(str)
    out = (df.groupby(strata, group_keys=False)
             .apply(lambda g: g.sample(frac=frac, random_state=seed)))
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def apply_vertical_minimization(df, level_label, seed):
    """Retain the top fraction of features ranked by mutual information
    with the target. `target` and `sex_female` are always retained and
    excluded from ranking (informed-minimization design, Ch. 3).
    """
    frac = LEVELS[level_label]
    protected = ["target", "sex_female"]
    if frac == 1.00:
        return df.copy()

    feats = [c for c in df.columns if c not in protected]
    mi = mutual_info_classif(df[feats], df["target"],
                             discrete_features=True, random_state=seed)
    ranking = pd.Series(mi, index=feats).sort_values(ascending=False)

    n_keep = max(1, int(len(feats) * frac))
    keep = ranking.head(n_keep).index.tolist()
    return df[keep + protected].copy()


def make_split(df, seed, test_size=0.30):
    """Stratified train/test split preserving class and subgroup
    proportions (stratified on target x sex_female)."""
    strata = df["target"].astype(str) + "_" + df["sex_female"].astype(str)
    train, test = train_test_split(df, test_size=test_size,
                                   stratify=strata, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)