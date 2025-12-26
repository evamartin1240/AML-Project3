# main.py

import os
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

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
    "stacking_no_xrfm": [],
    "stacking_with_xrfm": []
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

    # ---------------- Logistic Regression ----------------
    log_model, log_prob = train_logistic(Xtr, ytr, Xte)
    log_metrics = evaluate_model(yte, log_prob)

    all_results["logistic"].append({
        "metrics": log_metrics,
        "y_true": yte.tolist(),
        "y_prob": log_prob.tolist()
    })

    # ---------------- Random Forest ----------------
    rf_model, rf_prob = train_random_forest(Xtr, ytr, Xte)
    rf_metrics = evaluate_model(yte, rf_prob)

    all_results["random_forest"].append({
        "metrics": rf_metrics,
        "y_true": yte.tolist(),
        "y_prob": rf_prob.tolist()
    })

    # ---------------- Boosting ----------------
    gb_model, gb_prob = train_boosting(Xtr, ytr, Xte)
    gb_metrics = evaluate_model(yte, gb_prob)

    all_results["boosting"].append({
        "metrics": gb_metrics,
        "y_true": yte.tolist(),
        "y_prob": gb_prob.tolist()
    })

    # ---------------- xRFM ----------------
    xrfm_model, xrfm_prob = train_xrfm(Xtr, ytr, Xte)
    xrfm_metrics = evaluate_model(yte, xrfm_prob)

    all_results["xrfm"].append({
        "metrics": xrfm_metrics,
        "y_true": yte.tolist(),
        "y_prob": xrfm_prob.tolist()
    })

    # ---------------- STACKING (without xRFM) ----------------
    base_models_no_xrfm = {
        "logistic": train_logistic,
        "rf": train_random_forest,
        "gb": train_boosting,
    }

    stack_model, stack_prob = train_stacking(
        Xtr, ytr, Xte,
        base_models=base_models_no_xrfm,
        n_splits=3   # inner CV
    )

    stack_metrics = evaluate_model(yte, stack_prob)

    all_results["stacking_no_xrfm"].append({
        "metrics": stack_metrics,
        "y_true": yte.tolist(),
        "y_prob": stack_prob.tolist()
    })

    # ---------------- STACKING (with xRFM) ----------------
    base_models_with_xrfm = {
        "logistic": train_logistic,
        "rf": train_random_forest,
        "gb": train_boosting,
        "xrfm": train_xrfm,
    }

    stack_model_x, stack_prob_x = train_stacking(
        Xtr, ytr, Xte,
        base_models=base_models_with_xrfm,
        n_splits=3
    )

    stack_metrics_x = evaluate_model(yte, stack_prob_x)

    all_results["stacking_with_xrfm"].append({
        "metrics": stack_metrics_x,
        "y_true": yte.tolist(),
        "y_prob": stack_prob_x.tolist()
    })


# ===============================================================
# SAVE RESULTS
# ===============================================================

with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

total_time = time.time() - start_time
print(f"\n=== DONE — total time: {total_time/60:.2f} minutes ===")

