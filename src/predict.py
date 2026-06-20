"""
Prediction module for cyber attack detection system.
Loads trained model and generates predictions on new data.
"""

import os
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttackPredictor:
    """Generate predictions using trained model."""

    def __init__(self, model_dir='models'):
        """
        Initialize predictor and load model artifacts.

        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        self.load_all_artifacts()

    def load_all_artifacts(self):
        """Load all model and preprocessing artifacts."""
        try:
            model_path = os.path.join(self.model_dir, 'attack_detector.pkl')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            encoders_path = os.path.join(self.model_dir, 'encoders.pkl')

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoders = joblib.load(encoders_path)

            logger.info("All model artifacts loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model artifacts not found: {e}")
            raise

    def preprocess_features(self, features_dict):
        """
        Preprocess input features for prediction.

        Args:
            features_dict: Dictionary of feature values

        Returns:
            np.ndarray: Scaled features ready for prediction
        """
        # Convert dict to array in correct order
        feature_array = np.array([features_dict.get(f, 0) for f in self.feature_names])
        feature_array = feature_array.reshape(1, -1)

        # Scale features
        scaled_features = self.scaler.transform(feature_array)
        return scaled_features

    def predict(self, features_dict):
        """
        Predict attack probability for given features.

        Args:
            features_dict: Dictionary of feature values

        Returns:
            dict: Prediction result with label and confidence
        """
        try:
            scaled_features = self.preprocess_features(features_dict)
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            confidence = float(np.max(probabilities))

            result = {
                'prediction': int(prediction),
                'is_attack': prediction == 1,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            }

            logger.info(f"Prediction made: {result}")
            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(self, features_list):
        """
        Predict for multiple samples.

        Args:
            features_list: List of feature dictionaries

        Returns:
            list: List of predictions
        """
        results = []
        for features in features_list:
            results.append(self.predict(features))
        return results

    def explain_prediction(self, features_dict):
        """
        Provide explanation for prediction using feature importance.

        Args:
            features_dict: Dictionary of feature values

        Returns:
            dict: Explanation with important features
        """
        importances = self.model.feature_importances_
        top_indices = (-importances).argsort()[:5]

        explanation = {
            'top_features': [],
        }

        for idx in top_indices:
            if idx < len(self.feature_names):
                explanation['top_features'].append({
                    'feature': self.feature_names[idx],
                    'importance': float(importances[idx]),
                    'value': features_dict.get(self.feature_names[idx], 0)
                })

        return explanation

    def set_feature_names(self, feature_names):
        """Set feature names for prediction."""
        self.feature_names = feature_names

    @staticmethod
    def create_sample_prediction():
        """Create a sample prediction for testing."""
        sample_features = {
            'Flow Duration': 100000,
            'Total Fwd Packets': 50,
            'Total Backward Packets': 30,
            'Total Length of Fwd Packets': 5000,
            'Total Length of Bwd Packets': 3000,
            'Fwd Packet Length Max': 1500,
            'Fwd Packet Length Min': 50,
            'Fwd Packet Length Mean': 100,
        }
        return sample_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    try:
        predictor = AttackPredictor()
        sample = AttackPredictor.create_sample_prediction()
        result = predictor.predict(sample)
        print(f"Sample Prediction: {result}")
    except Exception as e:
        print(f"Error during prediction: {e}")
