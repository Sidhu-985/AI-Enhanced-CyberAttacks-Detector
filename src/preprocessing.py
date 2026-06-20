"""
Data preprocessing module for cyber attack detection system.
Handles loading, cleaning, and preparing data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_dataset(filepath):
    """
    Load CICIDS2017 or any CSV dataset.

    Args:
        filepath: Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def remove_missing_values(df):
    """
    Handle missing and infinite values in dataset.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    initial_rows = len(df)
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    logger.info(f"Removed {removed_rows} rows with missing/infinite values")
    return df


def encode_labels(df, label_column='Label'):
    """
    Encode categorical labels to numerical values.

    Args:
        df: Input DataFrame
        label_column: Name of label column

    Returns:
        tuple: (DataFrame with encoded labels, LabelEncoder)
    """
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])
    logger.info(f"Labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df, le


def encode_categorical_features(df, exclude_columns=None):
    """
    Encode categorical features using LabelEncoder.

    Args:
        df: Input DataFrame
        exclude_columns: Columns to exclude from encoding

    Returns:
        tuple: (DataFrame with encoded features, dict of encoders)
    """
    if exclude_columns is None:
        exclude_columns = []

    encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        if col not in exclude_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"Encoded feature: {col}")

    return df, encoders


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Testing features

    Returns:
        tuple: (Scaled X_train, Scaled X_test, StandardScaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split: Training {len(X_train)}, Testing {len(X_test)}")
    return X_train, X_test, y_train, y_test


def convert_to_binary_labels(y, normal_label='BENIGN'):
    """
    Convert multiclass labels to binary (Normal=0, Attack=1).

    Args:
        y: Original labels
        normal_label: Label representing normal traffic

    Returns:
        np.ndarray: Binary encoded labels
    """
    binary_labels = (y != normal_label).astype(int)
    logger.info(f"Converted to binary classification:")
    logger.info(f"  Normal (0): {(binary_labels == 0).sum()} samples")
    logger.info(f"  Attack (1): {(binary_labels == 1).sum()} samples")
    return binary_labels


def preprocess_pipeline(filepath, label_column='Label', test_size=0.2, binary=True, normal_label='BENIGN'):
    """
    Complete preprocessing pipeline for CICIDS2017 dataset.

    Args:
        filepath: Path to CSV file
        label_column: Name of label column
        test_size: Proportion of test set
        binary: Convert to binary classification (Normal vs Attack)
        normal_label: Label representing normal traffic

    Returns:
        dict: Preprocessing results and artifacts
    """
    logger.info("Starting preprocessing pipeline...")
    logger.info(f"Loading dataset from: {filepath}")

    # Load and clean
    df = load_dataset(filepath)
    initial_shape = df.shape
    df = remove_missing_values(df)
    logger.info(f"Shape after cleaning: {initial_shape} -> {df.shape}")

    # Separate features and labels
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset. Available: {df.columns.tolist()}")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    logger.info(f"Original label distribution:\n{y.value_counts()}")

    # Convert to binary classification if requested
    if binary:
        y_encoded = convert_to_binary_labels(y, normal_label)
    else:
        y_encoded = LabelEncoder().fit_transform(y)

    # Encode categorical features
    X, encoders = encode_categorical_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y_encoded, test_size=test_size
    )

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    logger.info("Preprocessing pipeline completed successfully")
    logger.info(f"Final dataset shapes:")
    logger.info(f"  X_train: {X_train_scaled.shape}")
    logger.info(f"  X_test: {X_test_scaled.shape}")
    logger.info(f"  y_train distribution: {np.bincount(y_train)}")
    logger.info(f"  y_test distribution: {np.bincount(y_test)}")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'encoders': encoders,
        'label_mapping': {0: 'Normal', 1: 'Attack'},
    }
