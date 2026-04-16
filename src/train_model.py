"""
Main script for training the churn prediction model.
"""

from data_processing import preprocess_data
from model import train_final_model
from utils import (
    plot_confusion_matrix,
    plot_distribution,
    plot_feature_importances,
    plot_grouped_distribution,
)


def run_eda(df):
    """Run Exploratory Data Analysis."""
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    plot_distribution(df, "Churn", "Churn Distribution")
    plot_grouped_distribution(df, "gender", "Churn", "Churn by Gender")
    plot_grouped_distribution(df, "Contract", "Churn", "Churn by Contract Type")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, columns = preprocess_data()

    # Run EDA
    from data_processing import clean_data, load_data

    df_clean = load_data()
    df_clean = clean_data(df_clean)
    run_eda(df_clean)

    # Train final model
    results = train_final_model(X_train, X_test, y_train, y_test, scaler, columns)

    # Plot results
    plot_confusion_matrix(
        y_test, results["metrics"]["y_pred"], "Optimized Gradient Boosting"
    )
    plot_feature_importances(results["feature_importances"])

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"\nBest Model: Optimized Gradient Boosting")
    print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"\nTop 5 Important Features:")
    for i, row in results["feature_importances"].head(5).iterrows():
        print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

    print("\nModel artifacts saved to 'models/' directory.")
    print("Plots saved to 'plots/' directory.")


if __name__ == "__main__":
    main()
