#!/bin/bash
set -Eeuo pipefail

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

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    echo "[ERROR] Do not run this script with sudo."
    echo "Run it as your normal user:"
    echo "  bash setup_orin.sh"
    exit 1
fi

# -----------------------------
# Helpers
# -----------------------------
ensure_user_owns_path() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Ensuring ownership of: $path"
        sudo chown -R "$USER_NAME:$USER_NAME" "$path" || true
    fi
}

safe_remove_path() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Removing existing path: $path"
        sudo rm -rf "$path"
    fi
}

# -----------------------------
# System packages
# -----------------------------
echo "Installing system dependencies..."
sudo apt-get update

sudo apt-get install -y \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  pkg-config \
  curl \
  portaudio19-dev \
  python3-pyaudio \
  libasound2-dev \
  flac \
  x11-xserver-utils

echo "System dependencies installed"

# -----------------------------
# Install Ollama if missing
# -----------------------------
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

sudo systemctl enable ollama
sudo systemctl start ollama

OLLAMA_SVC="/etc/systemd/system/ollama.service"
if [ -f "$OLLAMA_SVC" ]; then
    CHANGED=0

    if ! grep -q 'OLLAMA_NUM_GPU' "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_NUM_GPU=999"' "$OLLAMA_SVC"
        CHANGED=1
    fi
    if ! grep -q 'OLLAMA_FLASH_ATTENTION' "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_FLASH_ATTENTION=1"' "$OLLAMA_SVC"
        CHANGED=1
    fi
    if ! grep -q 'OLLAMA_KV_CACHE_TYPE' "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_KV_CACHE_TYPE=q8_0"' "$OLLAMA_SVC"
        CHANGED=1
    fi

    if [ "$CHANGED" -eq 1 ]; then
        sudo systemctl daemon-reload
        sudo systemctl restart ollama
        sleep 3
        echo "Ollama GPU environment configured"
    fi
fi

echo "Downloading models..."
ollama pull llama3.2
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# -----------------------------
# Clean + rebuild venv
# -----------------------------
VENV_DIR="$PROJECT_DIR/aura_env"

if [ -d "$VENV_DIR" ]; then
    echo "Existing aura_env detected"
    ensure_user_owns_path "$VENV_DIR"
    safe_remove_path "$VENV_DIR"
fi

echo "Creating fresh virtual environment..."
python3 -m venv "$VENV_DIR"

ensure_user_owns_path "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PIP_REQUIRE_VIRTUALENV=true
unset PIP_USER
unset PYTHONUSERBASE

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing Python requirements..."
python -m pip install --no-user -r "$PROJECT_DIR/requirements.txt"

echo "Installing Jetson-specific Python libraries..."
python -m pip install --no-user jetson-stats SpeechRecognition

# -----------------------------
# Create startup launcher
# -----------------------------
echo "Creating startup launcher..."
cat > "$PROJECT_DIR/start_aura.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

exec python agent/main.py
EOF

chmod +x "$PROJECT_DIR/start_aura.sh"
ensure_user_owns_path "$PROJECT_DIR/start_aura.sh"

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
After=network-online.target ollama.service
Wants=network-online.target
Requires=ollama.service

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
Exec=/bin/bash -lc 'sleep 3; OUTPUT=$(xrandr --query | awk "/ connected/{print \$1; exit}"); if [ -n "$OUTPUT" ]; then xrandr --output "$OUTPUT" --rotate left; fi'
X-GNOME-Autostart-enabled=true
EOF

sudo chown -R "$USER_NAME:$USER_NAME" "$USER_HOME/.config"

# -----------------------------
# Final ownership cleanup
# -----------------------------
ensure_user_owns_path "$PROJECT_DIR"
ensure_user_owns_path "$VENV_DIR"

# -----------------------------
# Done
# -----------------------------
echo ""
echo "===================================="
echo "Setup Complete"
echo "===================================="
echo ""
echo "What this now does:"
echo "1. Fixes old root-owned aura_env issues"
echo "2. Rebuilds aura_env cleanly"
echo "3. Installs all Python/system dependencies into the venv"
echo "4. Auto-starts agent/main.py on boot"
echo "5. Auto-applies /dev/ttyACM0 permissions"
echo "6. Auto-rotates display right on login"
echo ""
echo "Useful commands:"
echo "systemctl status $SERVICE_NAME"
echo "journalctl -u $SERVICE_NAME -f"
echo ""
echo "If you still hit permission issues, run this once:"
echo "sudo chown -R $USER_NAME:$USER_NAME \"$PROJECT_DIR\""
echo ""
echo "Reboot recommended:"
echo "sudo reboot"
echo ""