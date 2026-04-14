#!/bin/bash
set -e

echo "===================================="
echo " AURA Setup for Jetson Orin Nano"
echo "===================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
USER_HOME="$(eval echo "~$USER_NAME")"
SERVICE_NAME="aura-agent.service"

echo ""
echo "Project dir: $PROJECT_DIR"
echo "User: $USER_NAME"
echo "Home: $USER_HOME"
echo ""

# -----------------------------
# System packages
# -----------------------------
sudo apt-get update

sudo apt-get install -y \
  python3-pip \
  python3-venv \
  curl \
  portaudio19-dev \
  python3-pyaudio \
  flac \
  x11-xserver-utils

echo "System dependencies installed"

# -----------------------------
# Install Ollama if missing
# -----------------------------
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

sudo systemctl enable ollama
sudo systemctl start ollama

# Force Ollama to use all GPU layers and enable performance features.
#
# OLLAMA_NUM_GPU=999          — offload all model layers to the Jetson GPU
# OLLAMA_FLASH_ATTENTION=1    — use flash attention (reduces VRAM usage per call,
#                               keeps model in GPU when camera/YOLO also run)
# OLLAMA_KV_CACHE_TYPE=q8_0  — quantise KV cache fp16→int8 (halves KV VRAM;
#                               the single biggest reason Ollama falls back to
#                               CPU on unified-memory Jetsons)
OLLAMA_SVC="/etc/systemd/system/ollama.service"
if [ -f "$OLLAMA_SVC" ]; then
    CHANGED=0
    if ! grep -q "OLLAMA_NUM_GPU" "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_NUM_GPU=999"' "$OLLAMA_SVC"
        CHANGED=1
    fi
    if ! grep -q "OLLAMA_FLASH_ATTENTION" "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_FLASH_ATTENTION=1"' "$OLLAMA_SVC"
        CHANGED=1
    fi
    if ! grep -q "OLLAMA_KV_CACHE_TYPE" "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_KV_CACHE_TYPE=q8_0"' "$OLLAMA_SVC"
        CHANGED=1
    fi
    if [ "$CHANGED" -eq 1 ]; then
        sudo systemctl daemon-reload
        sudo systemctl restart ollama
        sleep 3
        echo "Ollama GPU environment configured (flash attention + q8_0 KV cache)"
    fi
fi

echo "Downloading models..."
ollama pull llama3.2
ollama pull llama3.2:1b        # lightweight intent-classification model (~2x faster)
ollama pull nomic-embed-text

# -----------------------------
# Create venv
# -----------------------------
if [ ! -d "$PROJECT_DIR/aura_env" ]; then
    python3 -m venv "$PROJECT_DIR/aura_env" --system-site-packages
fi

source "$PROJECT_DIR/aura_env/bin/activate"

pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

echo "Installing Jetson specific libraries"
pip install jetson-stats pyserial SpeechRecognition faster-whisper

# -----------------------------
# Create startup launcher
# -----------------------------
echo "Creating startup launcher..."
cat > "$PROJECT_DIR/start_aura.sh" <<EOF
#!/usr/bin/env bash
set -e

cd "$PROJECT_DIR"
source "$PROJECT_DIR/aura_env/bin/activate"

exec python agent/main.py
EOF

chmod +x "$PROJECT_DIR/start_aura.sh"

# -----------------------------
# Serial permissions via udev
# -----------------------------
echo "Creating udev rule for /dev/ttyACM0..."
sudo tee /etc/udev/rules.d/99-aura-serial.rules > /dev/null <<'EOF'
SUBSYSTEM=="tty", KERNEL=="ttyACM0", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# -----------------------------
# Systemd service
# -----------------------------
echo "Creating systemd service..."
sudo tee "/etc/systemd/system/$SERVICE_NAME" > /dev/null <<EOF
[Unit]
Description=AURA Jetson Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/start_aura.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=AURA_NUM_GPU=99
Environment=AURA_INTENT_MODEL=llama3.2:1b
Environment=AURA_GRAPH_EXTRACT=true
Environment=AURA_KEEP_ALIVE=2h
Environment=AURA_NUM_CTX=2048
Environment=AURA_MAX_CTX_CHARS=4000
Environment=AURA_TOP_K=4

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

# -----------------------------
# Display rotate helper
# -----------------------------
echo "Creating display rotation autostart..."
mkdir -p "$USER_HOME/.config/autostart"

cat > "$USER_HOME/.config/autostart/aura-rotate.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=AURA Rotate Display
Exec=/bin/bash -lc 'sleep 3; OUTPUT=$(xrandr --query | awk "/ connected/{print \$1; exit}"); if [ -n "$OUTPUT" ]; then xrandr --output "$OUTPUT" --rotate right; fi'
X-GNOME-Autostart-enabled=true
EOF

chown -R "$USER_NAME:$USER_NAME" "$USER_HOME/.config"

# -----------------------------
# Done
# -----------------------------
echo ""
echo "===================================="
echo "Setup Complete"
echo "===================================="
echo ""
echo "What this now does:"
echo "1. Creates/uses aura_env"
echo "2. Installs all Python/system dependencies"
echo "3. Auto-starts agent/main.py on boot"
echo "4. Auto-applies /dev/ttyACM0 permissions"
echo "5. Auto-rotates display right on login"
echo ""
echo "Useful commands:"
echo "systemctl status $SERVICE_NAME"
echo "journalctl -u $SERVICE_NAME -f"
echo ""
echo "Reboot recommended:"
echo "sudo reboot"
echo ""