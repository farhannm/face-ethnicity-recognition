#!/bin/bash

echo "Installing TensorFlow for macOS and dependencies..."

# Detect chip architecture
ARCH=$(uname -m)

# Install the appropriate TensorFlow version based on architecture
if [ "$ARCH" = "arm64" ]; then
  echo "Detected Apple Silicon (M1/M2/M3). Installing tensorflow-macos..."
  pip uninstall -y tensorflow tensorflow-estimator keras
  pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.0
else
  echo "Detected Intel processor. Installing standard TensorFlow..."
  pip uninstall -y tensorflow tensorflow-estimator keras
  pip install tensorflow==2.13.0
fi

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models/weights
mkdir -p utils

echo "Installation complete!"
echo "Run the app with: streamlit run app.py"