"""
Model training, evaluation, and optimization functions.
"""

import pandas as pd
from config import CV_FOLDS, PARAM_GRID, RANDOM_STATE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, print_metrics, save_model_artifacts


def get_base_models() -> dict:
    """
    Get dictionary of base models to train.

    Returns:
        Dictionary of model name to model instance
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_base_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train base models and collect metrics.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels

    Returns:
        Dictionary of model name to metrics
    """
    print("\n" + "=" * 50)
    print("TRAINING BASE MODELS")
    print("=" * 50)

    models = get_base_models()
    all_metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, name)
        print_metrics(metrics)
        all_metrics[name] = metrics

    return all_metrics


def tune_hyperparameters(X_train, y_train) -> object:
    """
    Tune hyperparameters for Gradient Boosting using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Best model from grid search
    """
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)

    print("Starting hyperparameter tuning...")

    gb_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {grid_search.best_params_}")

    return grid_search.best_estimator_


def train_final_model(X_train, X_test, y_train, y_test, scaler, columns) -> dict:
    """
    Train and evaluate the final optimized model.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        scaler: Fitted StandardScaler
        columns: List of feature column names

    Returns:
        Dictionary containing model and metrics
    """
    print("\n" + "=" * 50)
    print("TRAINING FINAL MODEL")
    print("=" * 50)

    # Train base models for comparison
    base_metrics = train_base_models(X_train, X_test, y_train, y_test)

    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train, y_train)

    # Evaluate optimized model
    print("\n--- Optimized Gradient Boosting ---")
    optimized_metrics = evaluate_model(
        best_model, X_test, y_test, "Optimized Gradient Boosting"
    )
    print_metrics(optimized_metrics)

    # Save model artifacts
    save_model_artifacts(best_model, scaler, columns)

    # Get feature importances
    feature_importances = (
        pd.DataFrame(
            {"feature": columns, "importance": best_model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": best_model,
        "metrics": optimized_metrics,
        "base_metrics": base_metrics,
        "feature_importances": feature_importances,
    }
