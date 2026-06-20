"""
Model training module for cyber attack detection system.
Trains Random Forest classifier and saves the model.
"""

import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import preprocess_pipeline
from src.utils import evaluate_model

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage cyber attack detection models."""

    def __init__(self, model_dir='models', random_state=42):
        """
        Initialize ModelTrainer.

        Args:
            model_dir: Directory to save models
            random_state: Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.encoders = None
        os.makedirs(model_dir, exist_ok=True)

    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=20):
        """
        Train Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth

        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info(f"Training Random Forest with {n_estimators} estimators...")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        self.model.fit(X_train, y_train)
        logger.info("Random Forest training completed")
        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Testing features
            y_test: Testing labels

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        return metrics

    def save_model(self, model_name='attack_detector.pkl'):
        """
        Save trained model to disk.

        Args:
            model_name: Filename for the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def save_preprocessing_artifacts(self, scaler, encoders):
        """
        Save preprocessing artifacts (scaler, encoders).

        Args:
            scaler: StandardScaler object
            encoders: Dictionary of LabelEncoders
        """
        self.scaler = scaler
        self.encoders = encoders

        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'encoders.pkl')

        joblib.dump(scaler, scaler_path)
        joblib.dump(encoders, encoders_path)

        logger.info(f"Preprocessing artifacts saved")

    def load_model(self, model_name='attack_detector.pkl'):
        """
        Load trained model from disk.

        Args:
            model_name: Filename of the model
        """
        model_path = os.path.join(self.model_dir, model_name)
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

    def load_preprocessing_artifacts(self):
        """Load preprocessing artifacts from disk."""
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'encoders.pkl')

        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)

        logger.info("Preprocessing artifacts loaded")

    def get_feature_importance(self, top_n=20):
        """
        Get top N important features.

        Args:
            top_n: Number of top features to return

        Returns:
            pd.DataFrame: Feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importances = self.model.feature_importances_
        indices = (-importances).argsort()[:top_n]

        return {
            'importances': importances[indices].tolist(),
            'indices': indices.tolist()
        }


def train_full_pipeline(dataset_path):
    """
    Complete training pipeline from raw data to saved model.

    Args:
        dataset_path: Path to CSV dataset
    """
    logger.info("Starting full training pipeline...")

    # Preprocessing
    preprocessed = preprocess_pipeline(dataset_path)

    # Training
    trainer = ModelTrainer()
    trainer.train_random_forest(
        preprocessed['X_train'],
        preprocessed['y_train'],
        n_estimators=100,
        max_depth=20
    )

    # Evaluation
    metrics = trainer.evaluate(preprocessed['X_test'], preprocessed['y_test'])
    logger.info(f"Model Metrics: {metrics}")

    # Save artifacts
    trainer.save_model()
    trainer.save_preprocessing_artifacts(preprocessed['scaler'], preprocessed['encoders'])

    logger.info("Training pipeline completed successfully")
    return trainer, metrics


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    dataset_path = 'data/cicids2017_sample.csv'

    if os.path.exists(dataset_path):
        trainer, metrics = train_full_pipeline(dataset_path)
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print(f"Dataset not found at {dataset_path}")
        print("Please download CICIDS2017 dataset and place it in the data folder")
