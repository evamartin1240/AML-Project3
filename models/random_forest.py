# src/models/random_forest.py

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
    X_test,
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42
):
    """
    Train a Random Forest classifier and return predicted probabilities.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",   # clave para el desbalance
        n_jobs=-1,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Probabilidad de la clase positiva
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_prob

