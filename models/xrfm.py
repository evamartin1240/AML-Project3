# models/xrfm.py

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from xrfm import xRFM


def _to_numpy(X):
    """Accept pandas or numpy and return numpy float32."""
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=np.float32)


def _to_numpy_y(y):
    """Accept pandas or numpy and return numpy float32 vector."""
    if hasattr(y, "values"):
        y = y.values
    return np.asarray(y, dtype=np.float32).reshape(-1)


def train_xrfm(
    X_train,
    y_train,
    X_test,
    val_size=0.2,
    device=None,
    tuning_metric="auc",
    random_state=42,
    **rfm_kwargs
):
    """
    Train xRFM with required internal validation split.
    Works with pandas DataFrame / Series or numpy arrays.
    """

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Internal stratified split (REQUIRED by xRFM)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state
    )
    tr_idx, val_idx = next(splitter.split(X_train, y_train))

    # Slice safely (pandas or numpy)
    if hasattr(X_train, "iloc"):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    else:
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # Convert to torch tensors
    X_tr_t   = torch.from_numpy(_to_numpy(X_tr)).to(device)
    y_tr_t   = torch.from_numpy(_to_numpy_y(y_tr)).to(device)
    X_val_t  = torch.from_numpy(_to_numpy(X_val)).to(device)
    y_val_t  = torch.from_numpy(_to_numpy_y(y_val)).to(device)
    X_test_t = torch.from_numpy(_to_numpy(X_test)).to(device)

    # Model
    model = xRFM(
        device=device,
        tuning_metric=tuning_metric,
        **rfm_kwargs
    )

    # Fit (xRFM REQUIRES validation set)
    model.fit(X_tr_t, y_tr_t, X_val_t, y_val_t)

    # Predict probabilities
    y_prob = model.predict_proba(X_test_t)

    # Torch → numpy if needed
    if hasattr(y_prob, "detach"):
        y_prob = y_prob.detach().cpu().numpy()

    # Ensure 1D prob of positive class
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    return model, y_prob
