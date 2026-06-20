"""
FastAPI backend for cyber attack detection system.
Provides REST API endpoints for predictions and system status.
"""

import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import AttackPredictor
from src.utils import calculate_attack_probability

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cyber Attack Detection API",
    description="AI-Enhanced System for Network Attack Detection",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
try:
    predictor = AttackPredictor(model_dir='models')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for attack prediction."""
    features: dict


class PredictionResponse(BaseModel):
    """Response model for attack prediction."""
    prediction: int
    is_attack: bool
    confidence: float
    attack_probability: float
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    message: str


class SystemStatus(BaseModel):
    """Response model for system status."""
    status: str
    model_available: bool
    prediction_capability: bool
    version: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check system health and model availability.

    Returns:
        HealthResponse: System status
    """
    status = "healthy" if predictor is not None else "unhealthy"
    model_loaded = predictor is not None

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        message="System operational" if model_loaded else "Model not loaded"
    )


@app.get("/status", response_model=SystemStatus)
def system_status():
    """
    Get detailed system status.

    Returns:
        SystemStatus: Detailed status information
    """
    return SystemStatus(
        status="operational",
        model_available=predictor is not None,
        prediction_capability=predictor is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_attack(request: PredictionRequest):
    """
    Predict if network traffic is an attack.

    Args:
        request: PredictionRequest with features

    Returns:
        PredictionResponse: Prediction result

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please load the model first."
        )

    try:
        # Make prediction
        result = predictor.predict(request.features)

        # Calculate attack probability
        attack_prob = calculate_attack_probability(
            np.array(result['probabilities'])
        )

        message = "⚠️ ATTACK DETECTED" if result['is_attack'] else "✓ Normal Traffic"

        return PredictionResponse(
            prediction=result['prediction'],
            is_attack=result['is_attack'],
            confidence=result['confidence'],
            attack_probability=attack_prob,
            message=message
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch")
def predict_batch(requests: list[PredictionRequest]):
    """
    Batch prediction for multiple samples.

    Args:
        requests: List of PredictionRequest

    Returns:
        list: List of predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    results = []
    for req in requests:
        try:
            result = predictor.predict(req.features)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            results.append({"error": str(e)})

    return results


@app.get("/model-info")
def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        dict: Model information
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": predictor.model.n_estimators,
        "max_depth": predictor.model.max_depth,
        "n_features": predictor.model.n_features_in_,
        "classes": predictor.model.classes_.tolist()
    }


@app.get("/")
def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Cyber Attack Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "status": "GET /status",
            "predict": "POST /predict",
            "batch_predict": "POST /predict-batch",
            "model_info": "GET /model-info",
            "docs": "/docs"
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
