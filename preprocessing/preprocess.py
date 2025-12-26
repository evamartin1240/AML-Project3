# src/data/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Load dataset (generic)
# ------------------------------------------------------------

def load_data(path, target):
    """
    Load dataset and separate features / target.
    """
    df = pd.read_csv(path)
    y = df[target]
    X = df.drop(columns=[target])

    return X, y


# ------------------------------------------------------------
# 2. Winsorization (optional, fit on train only)
# ------------------------------------------------------------

def fit_winsorizer(X_train, factor=1.5):
    limits = {}

    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        limits[col] = (lower, upper)

    return limits


def apply_winsorizer(X, limits):
    X_wins = X.copy()

    for col, (lower, upper) in limits.items():
        X_wins[col] = np.clip(X_wins[col], lower, upper)

    return X_wins


# ------------------------------------------------------------
# 3. Scaling (fit on train only)
# ------------------------------------------------------------

def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X, scaler):
    return pd.DataFrame(
        scaler.transform(X),
        index=X.index,
        columns=X.columns
    )


# ------------------------------------------------------------
# 4. Preprocess one CV fold (leakage-free)
# ------------------------------------------------------------

def preprocess_fold(
    X, y, train_idx, test_idx,
    winsorize=False
):
    """
    Preprocess data inside a CV fold.
    """

    # ---- split ----
    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    # ---- winsorization (optional) ----
    if winsorize:
        limits = fit_winsorizer(X_train)
        X_train = apply_winsorizer(X_train, limits)
        X_test  = apply_winsorizer(X_test, limits)
    else:
        limits = None

    # ---- scaling ----
    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_test  = apply_scaler(X_test, scaler)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "winsor_limits": limits
    }

