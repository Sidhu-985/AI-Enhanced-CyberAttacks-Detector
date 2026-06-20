"""
Utility functions for the cyber attack detection system.
Includes logging, metrics, and helper functions.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance with multiple metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    logger.info(f"Model Evaluation: {metrics}")
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """Get confusion matrix for binary/multiclass classification."""
    return confusion_matrix(y_true, y_pred)


def calculate_attack_probability(prediction_proba):
    """
    Calculate probability of attack based on model prediction probabilities.

    Args:
        prediction_proba: Array of probabilities from model

    Returns:
        float: Attack probability (0-1)
    """
    if len(prediction_proba.shape) == 1:
        return float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction_proba[0])
    return float(np.max(prediction_proba))


def log_prediction(features, prediction, confidence):
    """Log prediction details for monitoring."""
    logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
