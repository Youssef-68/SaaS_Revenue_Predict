# SaaS Revenue Intelligence Platform Setup

echo "  SaaS Revenue Intelligence Platform"
echo "  Setup & Initialization"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create necessary directories
echo "Creating directories..."
mkdir -p data model .streamlit logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Check data files
echo ""
echo "Checking data files..."
if [ -f "data/cleaned_data.csv" ]; then
    echo "cleaned_data.csv found"
    DATA_FILE="data/cleaned_data.csv"
elif [ -f "data/saas_50k_v1.csv" ]; then
    echo "saas_50k_v1.csv found"
    DATA_FILE="data/saas_50k_v1.csv"
else
    echo "No data file found in data/ directory"
    echo "   Please add your CSV file to the data/ folder"
    DATA_FILE=""
fi

# Train model if data exists
if [ -n "$DATA_FILE" ]; then
    echo ""
    echo "Training machine learning models..."
    echo "   This may take a few minutes..."
    python src/train.py
    echo "Model training complete!"
else
    echo ""
    echo "Skipping model training (no data file)"
    echo "   Run 'python src/train.py' after adding data"
fi

echo ""
echo "  Setup Complete!"
echo ""
echo "  To run the app:"
echo "    source venv/bin/activate"
echo "    streamlit run app.py"
echo ""
echo "  To train models:"
echo "    python src/train.py"
echo ""
echo "  To run notebooks:"
echo "    jupyter notebook notebooks/"