#!/bin/bash
# Quick setup and run script for Cyber Attack Detection System

set -e

echo "=================================="
echo "🛡️  Cyber Attack Detection System"
echo "Quick Setup & Run Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create virtual environment
echo -e "${BLUE}[1/5]${NC} Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Step 2: Activate virtual environment
echo -e "${BLUE}[2/5]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Step 3: Install dependencies
echo -e "${BLUE}[3/5]${NC} Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Dependencies installed"

# Step 4: Create sample dataset if needed
echo -e "${BLUE}[4/5]${NC} Preparing dataset..."
if [ ! -f "data/cicids2017_sample.csv" ]; then
    echo "Creating sample dataset..."
    python data_loader.py
    echo -e "${GREEN}✓${NC} Sample dataset created"
else
    echo -e "${GREEN}✓${NC} Dataset already exists"
fi

# Step 5: Train model if needed
echo -e "${BLUE}[5/5]${NC} Checking trained model..."
if [ ! -f "models/attack_detector.pkl" ]; then
    echo "Training model (this may take a few minutes)..."
    python -m src.train
    echo -e "${GREEN}✓${NC} Model trained and saved"
else
    echo -e "${GREEN}✓${NC} Model already trained"
fi

echo ""
echo "=================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=================================="
echo ""
echo -e "${YELLOW}To start the system:${NC}"
echo ""
echo "1. Terminal 1 - Start FastAPI backend:"
echo "   python -m backend.app"
echo ""
echo "2. Terminal 2 - Open frontend dashboard:"
echo "   cd frontend && python -m http.server 8080"
echo "   Then visit: http://localhost:8080"
echo ""
echo "3. API Documentation:"
echo "   http://localhost:8000/docs"
echo ""
