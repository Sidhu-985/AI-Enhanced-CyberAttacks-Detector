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
        Train Random Forest classifier with balanced class weights.

        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth

        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info(f"Training Random Forest with {n_estimators} estimators...")
        logger.info(f"  Config: max_depth={max_depth}, class_weight=balanced, n_jobs=-1")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
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

    def save_model(self, model_name='random_forest.pkl'):
        """
        Save trained model to disk.

        Args:
            model_name: Filename for the model (default: random_forest.pkl)
        """
        if self.model is None:
            raise ValueError("No model to save")

        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path

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


def train_full_pipeline(dataset_path, max_rows=100000, label_column='Label'):
    """
    Complete training pipeline from raw data to saved model.

    Args:
        dataset_path: Path to CSV dataset
        max_rows: Maximum rows to load (100000 for initial training)
        label_column: Name of label column
    """
    logger.info("="*60)
    logger.info("CYBER ATTACK DETECTION - ML TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Max Rows: {max_rows:,}")
    logger.info(f"Label Column: {label_column}")
    logger.info("="*60)

    # Load dataset with row limit
    logger.info("\n[1/4] Loading and preprocessing dataset...")
    import pandas as pd
    df = pd.read_csv(dataset_path, nrows=max_rows)
    logger.info(f"Loaded {len(df):,} rows from dataset")

    # Preprocessing
    preprocessed = preprocess_pipeline(
        dataset_path,
        label_column=label_column,
        binary=True,
        normal_label='BENIGN'
    )

    logger.info("\n[2/4] Training Random Forest Classifier...")
    logger.info(f"Training samples: {len(preprocessed['X_train']):,}")
    logger.info(f"Testing samples: {len(preprocessed['X_test']):,}")
    logger.info(f"Features: {len(preprocessed['feature_names'])}")

    # Training
    trainer = ModelTrainer()
    trainer.train_random_forest(
        preprocessed['X_train'],
        preprocessed['y_train'],
        n_estimators=100,
        max_depth=20
    )

    logger.info("\n[3/4] Evaluating model performance...")
    metrics = trainer.evaluate(preprocessed['X_test'], preprocessed['y_test'])

    # Print detailed metrics
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("="*60)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    y_pred = trainer.model.predict(preprocessed['X_test'])
    cm = confusion_matrix(preprocessed['y_test'], y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nCONFUSION MATRIX:")
    print(f"  True Negatives (Normal):  {tn:,}")
    print(f"  False Positives:          {fp:,}")
    print(f"  False Negatives:          {fn:,}")
    print(f"  True Positives (Attack):  {tp:,}")
    print("="*60)

    logger.info("\n[4/4] Saving model artifacts...")

    # Save artifacts
    trainer.save_model('random_forest.pkl')
    trainer.save_preprocessing_artifacts(preprocessed['scaler'], preprocessed['encoders'])
    logger.info(f"Model saved to: models/random_forest.pkl")
    logger.info(f"Scaler saved to: models/scaler.pkl")
    logger.info(f"Encoders saved to: models/encoders.pkl")

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY ✓")
    logger.info("="*60)

    return trainer, metrics


if __name__ == '__main__':
    # Example usage - Train on CICIDS2017 cleaned dataset
    logging.basicConfig(level=logging.INFO)

    dataset_path = 'data/cicids2017_cleaned.csv'

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("Please download CICIDS2017 dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print("\nOr download and place it in the data/ folder as 'cicids2017_cleaned.csv'")
        print("\nAlternatively, create a sample dataset:")
        print("  python data_loader.py")
    else:
        try:
            trainer, metrics = train_full_pipeline(
                dataset_path,
                max_rows=100000,
                label_column='Label'
            )
            print(f"\n✓ Model trained successfully!")
            print(f"✓ Metrics saved and model ready for predictions")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
