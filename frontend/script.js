// Cyber Attack Detection Dashboard - Frontend Logic

const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const resultsSection = document.getElementById('resultsSection');
const loadingContainer = document.getElementById('loadingContainer');
const errorContainer = document.getElementById('errorContainer');
const errorMessage = document.getElementById('errorMessage');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const statusMessage = document.getElementById('statusMessage');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkSystemHealth();
    setupEventListeners();
    setInterval(checkSystemHealth, 30000); // Check health every 30 seconds
});

// Setup event listeners
function setupEventListeners() {
    predictionForm.addEventListener('submit', handlePrediction);
}

// Check system health
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            statusDot.classList.remove('unhealthy', 'warning');
            statusDot.classList.add('healthy');
            statusText.textContent = 'System Healthy';
            statusMessage.textContent = '✓ Model loaded and ready for predictions';
        } else {
            statusDot.classList.remove('healthy');
            statusDot.classList.add('unhealthy');
            statusText.textContent = 'System Unhealthy';
            statusMessage.textContent = '⚠ Model not available';
        }
    } catch (error) {
        statusDot.classList.remove('healthy');
        statusDot.classList.add('unhealthy');
        statusText.textContent = 'Connection Error';
        statusMessage.textContent = `Cannot connect to API: ${error.message}`;
        console.error('Health check failed:', error);
    }
}

// Handle prediction submission
async function handlePrediction(event) {
    event.preventDefault();

    // Get form values
    const features = {
        'Flow Duration': parseFloat(document.getElementById('flowDuration').value),
        'Total Fwd Packets': parseFloat(document.getElementById('totalFwdPackets').value),
        'Total Backward Packets': parseFloat(document.getElementById('totalBwdPackets').value),
        'Total Length of Fwd Packets': parseFloat(document.getElementById('totalLenFwdPackets').value),
        'Total Length of Bwd Packets': parseFloat(document.getElementById('totalLenBwdPackets').value),
    };

    // Show loading state
    showLoading();
    hideError();
    hideResults();

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        showError(`Prediction failed: ${error.message}`);
        console.error('Prediction error:', error);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(result) {
    // Prediction result
    const resultLabel = document.getElementById('resultLabel');
    const resultIcon = document.getElementById('resultIcon');
    const messageBox = document.getElementById('resultMessage');

    if (result.is_attack) {
        resultLabel.textContent = '⚠️ ATTACK DETECTED';
        resultIcon.textContent = '🚨';
        messageBox.textContent = result.message;
        messageBox.style.borderLeftColor = 'var(--primary-color)';
    } else {
        resultLabel.textContent = '✓ NORMAL TRAFFIC';
        resultIcon.textContent = '✅';
        messageBox.textContent = result.message;
        messageBox.style.borderLeftColor = 'var(--success-color)';
    }

    // Confidence
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidencePercent = Math.round(result.confidence * 100);
    confidenceFill.style.width = confidencePercent + '%';
    confidenceValue.textContent = confidencePercent + '%';

    // Attack probability
    const probabilityValue = document.getElementById('probabilityValue');
    const probabilityPercent = Math.round(result.attack_probability * 100);
    probabilityValue.textContent = probabilityPercent + '%';

    // Update color based on probability
    const probabilityDisplay = document.getElementById('probabilityDisplay');
    if (result.attack_probability > 0.7) {
        probabilityDisplay.style.color = 'var(--primary-color)';
    } else if (result.attack_probability > 0.4) {
        probabilityDisplay.style.color = 'var(--warning-color)';
    } else {
        probabilityDisplay.style.color = 'var(--success-color)';
    }

    // Details
    document.getElementById('predictionClass').textContent = result.prediction === 1 ? 'Attack' : 'Normal';
    document.getElementById('isAttack').textContent = result.is_attack ? 'Yes ⚠️' : 'No ✓';
    document.getElementById('timestamp').textContent = new Date().toLocaleString();

    showResults();
}

// UI Helper Functions
function showLoading() {
    loadingContainer.style.display = 'block';
}

function hideLoading() {
    loadingContainer.style.display = 'none';
}

function showResults() {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = `❌ ${message}`;
    errorContainer.style.display = 'block';
}

function hideError() {
    errorContainer.style.display = 'none';
}

// Example function for future batch predictions
async function batchPredict(featuresList) {
    const requests = featuresList.map(features => ({ features }));

    try {
        const response = await fetch(`${API_BASE_URL}/predict-batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requests),
        });

        const results = await response.json();
        return results;
    } catch (error) {
        console.error('Batch prediction error:', error);
        throw error;
    }
}

// Get model info
async function getModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        const info = await response.json();
        console.log('Model Info:', info);
        return info;
    } catch (error) {
        console.error('Error fetching model info:', error);
    }
}
