# models/stacking.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def train_stacking(
    X_train,
    y_train,
    X_test,
    base_models,
    n_splits=3,
    random_state=42
):
    """
    Train stacking ensemble using out-of-fold predictions.

    Parameters
    ----------
    X_train, y_train : pandas DataFrame / Series
    X_test : pandas DataFrame
    base_models : dict
        { "name": train_function }
        where train_function(Xtr, ytr, Xval) -> (model, y_prob)
    n_splits : int
        Inner CV folds for OOF predictions
    """

    n_models = len(base_models)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Meta-features
    Z_train = np.zeros((n_train, n_models))
    Z_test = np.zeros((n_test, n_models))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    # --------------------------------------------------
    # Generate OOF predictions for each base model
    # --------------------------------------------------
    for m_idx, (name, train_fn) in enumerate(base_models.items()):
        print(f"\n[STACKING] Base model: {name}")

        test_preds_folds = []

        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            print(f"  Inner fold {fold_idx}/{n_splits}")

            if hasattr(X_train, "iloc"):
                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_val = X_train.iloc[val_idx]
            else:
                X_tr = X_train[tr_idx]
                y_tr = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else y_train[tr_idx]
                X_val = X_train[val_idx]


            # Train once per fold
            model, val_prob = train_fn(X_tr, y_tr, X_val)
            Z_train[val_idx, m_idx] = val_prob

            # Predict on test with same model
            _, test_prob = train_fn(X_tr, y_tr, X_test)
            test_preds_folds.append(test_prob)

        # Average test predictions across folds
        Z_test[:, m_idx] = np.mean(test_preds_folds, axis=0)

    # --------------------------------------------------
    # Meta-model (simple & calibrated)
    # --------------------------------------------------
    meta_model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=random_state
    )

    meta_model.fit(Z_train, y_train)

    y_prob = meta_model.predict_proba(Z_test)[:, 1]

    return meta_model, y_prob
