from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

def train_boosting(
    X_train,
    y_train,
    X_test,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
):

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_prob

