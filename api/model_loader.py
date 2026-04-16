import pickle
from pathlib import Path

# Resolve the path to the models directory relative to this file
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

# Module-level cache for loaded artifacts
_model = None
_scaler = None
_training_columns = None


def load_models():
    """
    Load the trained ML model, scaler, and training column names from disk.
    Models are cached in memory after the first load to avoid repeated I/O.

    Returns:
        tuple: (model, scaler, training_columns)
    """
    global _model, _scaler, _training_columns

    # Return cached models if already loaded
    if _model is not None and _scaler is not None and _training_columns is not None:
        return _model, _scaler, _training_columns

    model_path = MODEL_DIR / "churn_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    columns_path = MODEL_DIR / "training_columns.pkl"

    print(f"🔄 Loading model artifacts from {MODEL_DIR}...")

    try:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)

        with open(columns_path, "rb") as f:
            _training_columns = pickle.load(f)

        print("✅ Model artifacts loaded successfully!")

    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model file not found: {e.filename}. Please ensure models are trained and saved."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}")

    return _model, _scaler, _training_columns


def get_models():
    """
    Retrieve the loaded model artifacts. Loads them if not already in memory.

    Returns:
        tuple: (model, scaler, training_columns)
    """
    if _model is None:
        return load_models()
    return _model, _scaler, _training_columns
