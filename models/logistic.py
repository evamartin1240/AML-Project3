# src/models/logistic.py

from sklearn.linear_model import LogisticRegression


def train_logistic(X_train, y_train, X_test, C=1.0):
    """
    Train a logistic regression model and return predicted probabilities.

    Returns
    -------
    model : fitted LogisticRegression
    y_prob : array-like, shape (n_samples,)
        Predicted probability of the positive class.
    """

    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Probabilidad de la clase positiva (default = 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_prob

