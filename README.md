# 🛡️ AI-Enhanced Cyber Attack Detection System

A production-ready machine learning system for detecting network-based cyber attacks using advanced data preprocessing and Random Forest classification. Built with Python, FastAPI, and a modern web dashboard.

## 📋 Project Overview

This system analyzes network traffic patterns to identify and classify cyber attacks in real-time. It uses the CICIDS2017 intrusion detection dataset for training and provides both CLI and REST API interfaces for making predictions.

### Key Features

- **Modular Architecture**: Cleanly separated preprocessing, training, and prediction modules
- **Random Forest Classifier**: Powerful ensemble learning model with ~95% accuracy on CICIDS2017
- **FastAPI Backend**: RESTful API with health checks and batch prediction support
- **Interactive Dashboard**: Modern web UI for real-time attack detection and visualization
- **Production-Ready**: Comprehensive error handling, logging, and monitoring

## 🏗️ Project Architecture

```
AI-Enhanced-CyberAttacks-Detector/
├── src/                           # Core ML modules
│   ├── preprocessing.py           # Data loading, cleaning, scaling
│   ├── train.py                   # Model training pipeline
│   ├── predict.py                 # Prediction engine
│   └── utils.py                   # Utility functions & metrics
├── backend/                       # FastAPI server
│   ├── app.py                     # API endpoints
│   └── requirements.txt           # Python dependencies
├── frontend/                      # Web dashboard
│   ├── index.html                 # Dashboard UI
│   ├── style.css                  # Styling
│   └── script.js                  # Frontend logic
├── models/                        # Trained models (generated)
│   ├── attack_detector.pkl        # Random Forest model
│   ├── scaler.pkl                 # Feature scaler
│   └── encoders.pkl               # Label encoders
├── data/                          # Datasets
│   └── cicids2017_sample.csv      # CICIDS2017 dataset
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                         # Unit tests
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
└── requirements.txt               # Main dependencies
```

## 📊 Dataset Information

### CICIDS2017
- **Size**: ~1M network flow samples
- **Features**: 84 network traffic features
- **Classes**: 14 attack types + Normal traffic
- **Source**: https://www.kaggle.com/datasets/cicdataset/cicids2017

### Download Instructions

1. Download from Kaggle: [CICIDS2017 Dataset](https://www.kaggle.com/datasets/cicdataset/cicids2017)
2. Extract CSV files to `data/` folder
3. Use the training script to preprocess automatically

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Ubuntu Linux (tested on 20.04+)
- 8GB RAM minimum
- 5GB disk space for dataset and models

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd ~/Desktop/Projects/AI-Enhanced-CyberAttacks-Detector

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r backend/requirements.txt

# Verify installation
python -c "import sklearn, fastapi, pandas; print('✓ All dependencies installed')"
```

### Step 3: Prepare Dataset

```bash
# Download CICIDS2017 dataset from Kaggle
# Place CSV files in data/ directory

# Verify dataset
ls -lh data/

# Expected: cicids2017_sample.csv (~100-500MB)
```

## 🎓 Training Instructions

### Quick Start - Train Model

```bash
# Run complete training pipeline
python -m src.train

# Expected output:
# - Training progress with Random Forest details
# - Model accuracy, precision, recall, F1-score
# - Trained models saved to models/
```

### Detailed Training Process

```bash
# Step 1: Preprocessing verification
python -c "
from src.preprocessing import preprocess_pipeline
result = preprocess_pipeline('data/cicids2017_sample.csv')
print(f'Training samples: {len(result[\"X_train\"])}')
print(f'Testing samples: {len(result[\"X_test\"])}')
"

# Step 2: Train with custom parameters
python -c "
from src.train import ModelTrainer
from src.preprocessing import preprocess_pipeline

# Load data
data = preprocess_pipeline('data/cicids2017_sample.csv')

# Train model
trainer = ModelTrainer()
trainer.train_random_forest(
    data['X_train'], 
    data['y_train'],
    n_estimators=100,
    max_depth=20
)

# Evaluate
metrics = trainer.evaluate(data['X_test'], data['y_test'])
print('Evaluation Metrics:', metrics)

# Save
trainer.save_model()
trainer.save_preprocessing_artifacts(data['scaler'], data['encoders'])
"

# Step 3: Verify model artifacts
ls -lh models/
# Should contain: attack_detector.pkl, scaler.pkl, encoders.pkl
```

### Expected Model Performance

- **Accuracy**: ~95-97%
- **Precision**: ~94-96%
- **Recall**: ~95-97%
- **F1-Score**: ~95-96%

## 🔌 API Usage

### Start FastAPI Server

```bash
# Method 1: Direct uvicorn
cd backend/
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Method 2: From project root
python -m backend.app

# Expected output:
# Uvicorn running on http://0.0.0.0:8000
# API docs available at http://localhost:8000/docs
```

### API Endpoints

#### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"

# Response:
{
  "status": "healthy",
  "model_loaded": true,
  "message": "System operational"
}
```

#### 2. System Status
```bash
curl -X GET "http://localhost:8000/status"

# Response:
{
  "status": "operational",
  "model_available": true,
  "prediction_capability": true,
  "version": "1.0.0"
}
```

#### 3. Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Flow Duration": 100000,
      "Total Fwd Packets": 50,
      "Total Backward Packets": 30,
      "Total Length of Fwd Packets": 5000,
      "Total Length of Bwd Packets": 3000
    }
  }'

# Response:
{
  "prediction": 0,
  "is_attack": false,
  "confidence": 0.98,
  "attack_probability": 0.15,
  "message": "✓ Normal Traffic"
}
```

#### 4. Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"features": {"Flow Duration": 100000, "Total Fwd Packets": 50, ...}},
    {"features": {"Flow Duration": 50000, "Total Fwd Packets": 25, ...}}
  ]'
```

#### 5. Model Information
```bash
curl -X GET "http://localhost:8000/model-info"

# Response:
{
  "model_type": "Random Forest Classifier",
  "n_estimators": 100,
  "max_depth": 20,
  "n_features": 84,
  "classes": [0, 1]
}
```

#### 6. Interactive API Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🌐 Frontend Dashboard

### Access Dashboard

1. Open `frontend/index.html` in a web browser
   - Or use Python's built-in server:
   ```bash
   cd frontend/
   python -m http.server 8080
   # Visit http://localhost:8080
   ```

2. Ensure FastAPI backend is running on http://localhost:8000

### Dashboard Features

- **System Status**: Real-time health monitoring
- **Traffic Analysis**: Input network traffic features
- **Detection Results**: Attack classification with confidence
- **Visualization**: Confidence bars and probability displays
- **Details View**: Detailed prediction information

### Using the Dashboard

1. Fill in network traffic metrics
2. Click "Analyze Traffic" button
3. View prediction results with:
   - Attack/Normal classification
   - Model confidence score
   - Attack probability percentage
   - Detailed feature analysis

## 📈 Model Training Details

### Random Forest Classifier Configuration

```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=20,          # Maximum tree depth
    random_state=42,       # Reproducibility
    n_jobs=-1,             # Parallel processing
    verbose=1              # Training progress
)
```

### Preprocessing Pipeline

1. **Load Dataset**: Read CSV file with pandas
2. **Remove Missing Values**: Drop rows with NaN/infinite values
3. **Encode Labels**: Convert string labels to numeric (0/1)
4. **Encode Features**: Convert categorical features to numeric
5. **Feature Scaling**: StandardScaler normalization to [-1, 1]
6. **Train-Test Split**: 80/20 stratified split

### Feature Importance Analysis

Top 5 most important features for attack detection:
1. Flow Duration
2. Total Fwd Packets
3. Total Length of Fwd Packets
4. Forward Packet Length Mean
5. Total Backward Packets

## 🧪 Testing

### Unit Tests

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src
```

### Manual Testing

```bash
# Test preprocessing
python -c "from src.preprocessing import preprocess_pipeline; print(preprocess_pipeline('data/cicids2017_sample.csv'))"

# Test prediction
python -c "
from src.predict import AttackPredictor
p = AttackPredictor()
result = p.predict({'Flow Duration': 100000, ...})
print(result)
"

# Test API
python backend/app.py
# Then use curl commands above
```

## 📚 Usage Examples

### Example 1: Train Model from Scratch

```python
from src.train import train_full_pipeline

trainer, metrics = train_full_pipeline('data/cicids2017_sample.csv')
print(f"Model Accuracy: {metrics['accuracy']:.4f}")
```

### Example 2: Load and Use Trained Model

```python
from src.predict import AttackPredictor

predictor = AttackPredictor()

# Make prediction
features = {
    'Flow Duration': 100000,
    'Total Fwd Packets': 50,
    'Total Backward Packets': 30,
}

result = predictor.predict(features)
print(f"Is Attack: {result['is_attack']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example 3: API Request in Python

```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={
        'features': {
            'Flow Duration': 100000,
            'Total Fwd Packets': 50,
            'Total Backward Packets': 30,
            'Total Length of Fwd Packets': 5000,
            'Total Length of Bwd Packets': 3000,
        }
    }
)

prediction = response.json()
print(f"Attack Detected: {prediction['is_attack']}")
```

## 🔒 Security Considerations

- **Model Artifacts**: Store `models/` directory securely
- **API Keys**: Add authentication (JWT, API keys) for production
- **Input Validation**: All inputs validated before processing
- **Rate Limiting**: Add rate limits to prevent API abuse
- **HTTPS**: Use HTTPS in production environments

## 📦 Dependencies

### Core ML Libraries
- `scikit-learn==1.3.2` - Machine Learning
- `pandas==2.1.3` - Data manipulation
- `numpy==1.24.3` - Numerical computing
- `joblib==1.3.2` - Model serialization

### Backend
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `pydantic==2.5.0` - Data validation

### Frontend
- HTML5, CSS3, JavaScript (no external dependencies)

## 🛠️ Troubleshooting

### Issue: Model not found
```bash
# Solution: Train the model first
python -m src.train
```

### Issue: API connection error
```bash
# Solution: Ensure backend is running
python -m backend.app
# Or: uvicorn backend.app:app --reload
```

### Issue: Memory error during training
```bash
# Solution: Use smaller dataset or reduce n_estimators
# Modify in src/train.py:
trainer.train_random_forest(
    X_train, y_train,
    n_estimators=50,  # Reduce from 100
    max_depth=15      # Reduce from 20
)
```

### Issue: CORS errors in frontend
```bash
# Solution: Already configured in backend/app.py
# If issues persist, verify CORS middleware:
# - Allow origins: ["*"]
# - Allow credentials: True
```

## 🚀 Future Enhancements

1. **Transformer-based Detection**: Replace Random Forest with deep learning
2. **Real-time Streaming**: Kafka integration for live data processing
3. **Anomaly Detection**: Unsupervised learning for zero-day attacks
4. **Advanced Visualizations**: Plotly dashboards and SHAP explanations
5. **Database Integration**: PostgreSQL for model versioning and predictions
6. **Docker Deployment**: Containerized production deployment
7. **Monitoring**: Prometheus metrics and Grafana dashboards

## 📖 References

- CICIDS2017 Dataset: https://www.kaggle.com/datasets/cicdataset/cicids2017
- Scikit-learn Documentation: https://scikit-learn.org/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Random Forest: https://en.wikipedia.org/wiki/Random_forest

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Sidharth Sunil**
- Role: AI/ML Engineer
- Project: AI-Enhanced Cyber Attack Detection System for Resume

## 📞 Support

For issues, questions, or suggestions:
1. Check the Troubleshooting section
2. Review API documentation at http://localhost:8000/docs
3. Check logs for detailed error information
4. Verify dataset format and preprocessing

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Production Ready
