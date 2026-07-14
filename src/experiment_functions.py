"""Training, metrics, and run execution (Osilesi dissertation, Ch. 3).

Builds on manipulation_functions.py. Three classifiers with fixed
hyperparameters (no tuning by design); seven dependent variables
computed on the test partition; W&B-logged run wrapper.

Fairness sign convention: female minus male. Negative
equal_opportunity_diff means the female subgroup has the lower TPR.
Logistic regression is wrapped with StandardScaler (train-fit only)
for lbfgs convergence; RF and XGBoost are scale-invariant and unscaled.

W&B runs in offline mode (network removed from the execution critical
path after repeated online-init stalls during the 960-run batch); runs
are written locally under ./wandb and synced afterward in one step with
`wandb sync --sync-all` from the JupyterLab terminal.
"""

import numpy as np
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef)

from manipulation_functions import (apply_imbalance,
                                    apply_horizontal_minimization,
                                    apply_vertical_minimization,
                                    make_split)


def train_model(train_df, classifier_name, seed):
    """Fit one of the three classifiers on the training partition.

    Hyperparameters are fixed across all conditions by design; only
    the seed varies for stochastic classifiers.
    """
    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    if classifier_name == "logistic_regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed))
    elif classifier_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=seed, n_jobs=-1)
    elif classifier_name == "xgboost":
        model = XGBClassifier(n_estimators=100, max_depth=6,
                              learning_rate=0.3, random_state=seed,
                              eval_metric="logloss", n_jobs=-1)
    else:
        raise ValueError(classifier_name)

    model.fit(X, y)
    return model


def compute_metrics(model, test_df):
    """Compute all seven dependent variables on the test set,
    plus subgroup diagnostics."""
    X = test_df.drop(columns=["target"])
    y = test_df["target"].values
    pred = model.predict(X)
    female = test_df["sex_female"].values == 1

    m = {
        "accuracy":          accuracy_score(y, pred),
        "minority_recall":   recall_score(y, pred, pos_label=1),
        "macro_f1":          f1_score(y, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc":               matthews_corrcoef(y, pred),
    }

    def tpr(mask):
        pos = (y == 1) & mask
        return np.nan if pos.sum() == 0 else pred[pos].mean()

    tpr_f, tpr_m = tpr(female), tpr(~female)
    m["equal_opportunity_diff"] = tpr_f - tpr_m
    m["fnr_diff"] = (1 - tpr_f) - (1 - tpr_m)

    m["n_test"] = len(y)
    m["n_female_pos_test"] = int(((y == 1) & female).sum())
    return m


def run_experiment(df, dataset_name, classifier_name, imbalance,
                   min_level, min_type, seed, base_n=15000,
                   project="dissertation-rq-experiments"):
    """Execute one complete run: manipulate, split, train, score, log."""
    config = dict(dataset=dataset_name, classifier=classifier_name,
                  imbalance=imbalance, min_level=min_level,
                  min_type=min_type, seed=seed)

    run = wandb.init(project=project, config=config,
                     name=f"{dataset_name}-{classifier_name}-{imbalance}"
                          f"-{min_type}{min_level}-s{seed}",
                     reinit=True, mode="offline")
    try:
        d = apply_imbalance(df, imbalance, seed, base_n=base_n)
        if min_type == "horizontal":
            d = apply_horizontal_minimization(d, min_level, seed)
        else:
            d = apply_vertical_minimization(d, min_level, seed)
        train, test = make_split(d, seed)

        model = train_model(train, classifier_name, seed)
        metrics = compute_metrics(model, test)

        wandb.log(metrics)
        result = {**config, **metrics, "status": "ok"}
    except Exception as e:
        result = {**config, "status": f"error: {e}"}
    finally:
        run.finish()
    return result