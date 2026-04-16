import os

import numpy as np
import pandas as pd
from config import BINARY_COLS, DATA_PATH, RANDOM_STATE, TARGET_COL, TEST_SIZE
from fsspec.utils import T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Load data from CSV file
    Aegs:
        file_path:Path to CSV file
    Returns:
        Dataframe with loaded data"""

    df = pd.read_csv(file_path)
    print(df.shape)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic data exploration information.

    Args:
        df: DataFrame to explore
    """
    """
       Print basic data exploration information.

       Args:
           df: DataFrame to explore
       """
    print("\n" + "=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nData types and non-null values:")
    print(df.info())

    print(f"\nMissing values:")
    print(df.isnull().sum())

    print(f"\nSummary statistics (numeric):")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    print(f"\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Fill missing TotalCharges with median
    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

    # Drop customerID as it's not useful for prediction
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    print(f"\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


def encode_features(df: pd.date_range) -> pd.DataFrame:
    print("\n" + "=" * 50)
    print("FEATURE ENCODING")
    print("=" * 50)
    df = df.copy()
    le = LabelEncoder()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, drop_first=True)
    print(f"Data shape after feature engineering: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    if TARGET_COL in df.columns:
        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]
    else:
        X = df
        y = None

    columns = X.columns.to_list()

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, ss, columns

    print(f"Features shape: {X_scaled.shape}")
    return X_scaled, y, ss, columns


def preprocess_data(filepath: str = DATA_PATH):
    """Complete preprocessing pipeline.

    Args:
        filepath: Path to the raw data
         is_training: If True, returns train/test split

    Returns:
        If training: (X_train, X_test, y_train, y_test, scaler, columns)
        If inference: (X_scaled, y, scaler, columns)
    """
    df = load_data(filepath)
    explore_data(df)
    df = clean_data(df)
    df = encode_features(df)
    return prepare_features(df)


preprocess_data()
