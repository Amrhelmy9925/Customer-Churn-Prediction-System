"""
Utility functions for model saving, loading, and evaluation.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def create_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def save_pickle(obj: object, filepath: str) -> None:
    """Save object to pickle file."""
    directory = os.path.dirname(filepath)
    if directory:
        create_directory(directory)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_pickle(filepath: str) -> object:
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_model_artifacts(model, scaler, columns) -> None:
    """Save model, scaler, and training columns."""
    create_directory("models")
    save_pickle(model, "models/churn_model.pkl")
    save_pickle(scaler, "models/scaler.pkl")
    save_pickle(columns, "models/training_columns.pkl")
    print("All model artifacts saved successfully.")


def load_model_artifacts():
    """Load model, scaler, and training columns."""
    model = load_pickle("models/churn_model.pkl")
    scaler = load_pickle("models/scaler.pkl")
    columns = load_pickle("models/training_columns.pkl")
    return model, scaler, columns


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics."""
    print(f"\n--- {metrics['model_name']} ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")


def save_plot(fig, filename: str) -> None:
    """Save plot to file."""
    create_directory("plots")
    filepath = f"plots/{filename}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved: {filepath}")


def plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    save_plot(fig, "confusion_matrix")


def plot_feature_importances(feature_importances, top_n: int = 15) -> None:
    """Plot and save feature importances."""
    features = feature_importances["feature"].values[:top_n].tolist()
    importances = feature_importances["importance"].values[:top_n].tolist()

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = range(len(features))
    ax.barh(y_pos, importances, color="steelblue")
    ax.set_yticks(y_pos, features)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances")

    save_plot(fig, "feature_importances")


def plot_distribution(data, column, title) -> None:
    """Plot and save value counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data[column].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_plot(fig, f"distribution_{column.lower()}")


def plot_grouped_distribution(data, group_col, target_col, title) -> None:
    """Plot and save grouped value counts."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Get value counts grouped by target
    df_grouped = data.groupby([group_col, target_col]).size().unstack(fill_value=0)
    df_grouped.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=target_col)
    plt.tight_layout()

    save_plot(fig, f"churn_by_{group_col.lower()}")
