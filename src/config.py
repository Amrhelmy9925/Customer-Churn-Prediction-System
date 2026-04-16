"""
Configuration constants for the churn prediction project.
"""

# Data paths
DATA_PATH = r"data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = "models"

# Model files
MODEL_FILE = "models/churn_model.pkl"
SCALER_FILE = "models/scaler.pkl"
COLUMNS_FILE = "models/training_columns.pkl"

# Binary columns for label encoding
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Target column
TARGET_COL = "Churn"

# Test size for train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparameter grid for Gradient Boosting
PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
}

# Cross-validation folds
CV_FOLDS = 3

# Top N features to display
TOP_N_FEATURES = 15
