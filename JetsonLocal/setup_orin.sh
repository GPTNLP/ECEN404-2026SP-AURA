#!/bin/bash
set -e

echo "===================================="
echo " AURA Setup for Jetson Orin Nano"
echo "===================================="

# Update system
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
python3-pip \
python3-venv \
curl \
portaudio19-dev \
python3-pyaudio \
flac

echo "System dependencies installed"

# Install Ollama if missing
if ! command -v ollama &> /dev/null
then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama
sudo systemctl enable ollama
sudo systemctl start ollama

echo "Downloading models..."

ollama pull llama3.2
ollama pull nomic-embed-text

# Create virtual environment
if [ ! -d "aura_env" ]; then
    python3 -m venv aura_env --system-site-packages
fi

source aura_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

echo "Installing Jetson specific libraries"

pip install jetson-stats pyserial SpeechRecognition faster-whisper

echo ""
echo "Setup Complete"
echo ""
echo "To start AURA:"
echo ""
echo "source aura_env/bin/activate"
echo "cd agent"
echo "python main.py"
echo ""
echo "Then open browser:"
echo "http://JETSON-IP:8000"
echo ""