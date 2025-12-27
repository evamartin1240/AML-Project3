import os
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from preprocessing.preprocess import load_data, preprocess_fold
from models.logistic import train_logistic
from models.random_forest import train_random_forest
from models.boosting import train_boosting
from models.xrfm import train_xrfm
from models.stacking import train_stacking

from evaluation.metrics import evaluate_model

# ===============================================================
# CONFIG
# ===============================================================

DATA_PATH = "data/data.csv"
TARGET = "default"
N_SPLITS = 5
RANDOM_SEED = 42
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

FOLDS_PATH = os.path.join(RESULTS_DIR, "cv_folds.json")
outer_folds = []

# ===============================================================
# LOAD DATA
# ===============================================================

X, y = load_data(DATA_PATH, TARGET)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

# container for results
all_results = {
    "logistic": [],
    "random_forest": [],
    "boosting": [],
    "xrfm": [],
    "stacking": []
}


start_time = time.time()

# ===============================================================
# LOOP OVER FOLDS
# ===============================================================

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")

    fold_data = preprocess_fold(
        X, y,
        train_idx, test_idx,
        winsorize=False
    )

    Xtr = fold_data["X_train"]
    ytr = fold_data["y_train"]
    Xte = fold_data["X_test"]
    yte = fold_data["y_test"]

    outer_folds.append({
        "fold": fold_idx,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist()
    })

    # ---------------- Logistic Regression ----------------
    t0 = time.time()
    log_model, log_prob = train_logistic(Xtr, ytr, Xte)
    log_time = time.time() - t0
    log_metrics = evaluate_model(yte, log_prob)

    all_results["logistic"].append({
        "metrics": log_metrics,
        "y_true": yte.tolist(),
        "y_prob": log_prob.tolist(),
        "time_sec": log_time
    })

    # ---------------- Random Forest ----------------
    t0 = time.time()
    rf_model, rf_prob = train_random_forest(Xtr, ytr, Xte)
    rf_time = time.time() - t0
    rf_metrics = evaluate_model(yte, rf_prob)

    all_results["random_forest"].append({
        "metrics": rf_metrics,
        "y_true": yte.tolist(),
        "y_prob": rf_prob.tolist(),
        "time_sec": rf_time
    })

    # ---------------- Boosting ----------------
    t0 = time.time()
    gb_model, gb_prob = train_boosting(Xtr, ytr, Xte)
    gb_time = time.time() - t0
    gb_metrics = evaluate_model(yte, gb_prob)

    all_results["boosting"].append({
        "metrics": gb_metrics,
        "y_true": yte.tolist(),
        "y_prob": gb_prob.tolist(),
        "time_sec": gb_time
    })

    # ---------------- xRFM ----------------
    t0 = time.time()
    xrfm_model, xrfm_prob = train_xrfm(Xtr, ytr, Xte)
    xrfm_time = time.time() - t0
    xrfm_metrics = evaluate_model(yte, xrfm_prob)

    all_results["xrfm"].append({
        "metrics": xrfm_metrics,
        "y_true": yte.tolist(),
        "y_prob": xrfm_prob.tolist(),
        "time_sec": xrfm_time
    })

    # ---------------- STACKING ----------------
    base_models = {
        "logistic": train_logistic,
        "rf": train_random_forest,
        "gb": train_boosting,
        "xrfm": train_xrfm,
    }

    t0 = time.time()
    stack_model_x, stack_prob_x = train_stacking(
        Xtr, ytr, Xte,
        base_models=base_models,
        n_splits=3
    )
    stack_time_x = time.time() - t0

    stack_metrics_x = evaluate_model(yte, stack_prob_x)

    all_results["stacking"].append({
        "metrics": stack_metrics_x,
        "y_true": yte.tolist(),
        "y_prob": stack_prob_x.tolist(),
        "time_sec": stack_time_x
    })

# ===============================================================
# SAVE RESULTS
# ===============================================================

with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

total_time = time.time() - start_time
print(f"\n=== DONE — total time: {total_time/60:.2f} minutes ===")

with open(FOLDS_PATH, "w") as f:
    json.dump(outer_folds, f, indent=2)

print(f"Saved CV folds to: {FOLDS_PATH}")

# ===============================================================
# Save metrics tables

RESULTS_PATH = "results/cv_results.json"

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

rows = []

for model_name, folds in results.items():
    for fold_idx, fold_data in enumerate(folds, start=1):
        metrics = fold_data["metrics"]
        time_sec = fold_data.get("time_sec", None)
        row = {
            "model": model_name,
            "fold": fold_idx,
        }
        row.update(metrics)
        if time_sec is not None:
            row["time_sec"] = time_sec
        rows.append(row)

df_folds = pd.DataFrame(rows)
# Format with 2 significant decimals
df_folds_fmt = df_folds.applymap(lambda x: f"{x:.2g}" if isinstance(x, (int, float)) else x)
df_folds_fmt.to_csv("results/metrics_by_fold.csv", index=False)

metrics_cols = [
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
    "f1",
    "balanced_accuracy",
    "time_sec",
]

df_summary = (
    df_folds
    .groupby("model")[metrics_cols]
    .agg(["mean", "std"])
)

# Format with 2 significant decimals
df_summary = df_summary.applymap(lambda x: f"{x:.2g}" if isinstance(x, (int, float)) else x)
df_summary.to_csv("results/metrics_summary.csv")