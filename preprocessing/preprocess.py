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

from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_fold(
    X, y, train_idx, test_idx,
    winsorize=False
):
    """
    Preprocess data inside a CV fold with proper categorical handling.
    """

    # --------------------
    # split
    # --------------------
    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test  = y.iloc[test_idx].copy()

    # --------------------
    # categorical handling
    # --------------------
    nominal_cats = ['SEX', 'EDUCATION', 'MARRIAGE']
    num_cols = [c for c in X.columns if c not in nominal_cats]

    # --------------------
    # winsorization (numeric only)
    # --------------------
    if winsorize:
        limits = fit_winsorizer(X_train[num_cols])
        X_train[num_cols] = apply_winsorizer(X_train[num_cols], limits)
        X_test[num_cols]  = apply_winsorizer(X_test[num_cols], limits)
    else:
        limits = None

    # --------------------
    # scaling (numeric only)
    # --------------------
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num  = scaler.transform(X_test[num_cols])

    # --------------------
    # one-hot encoding (nominal only)
    # --------------------
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = ohe.fit_transform(X_train[nominal_cats])
    X_test_cat  = ohe.transform(X_test[nominal_cats])

    # --------------------
    # concatenate
    # --------------------
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_test_final  = np.hstack([X_test_num, X_test_cat])

    return {
        "X_train": X_train_final,
        "y_train": y_train,
        "X_test": X_test_final,
        "y_test": y_test,
        "scaler": scaler,
        "ohe": ohe,
        "winsor_limits": limits
    }
