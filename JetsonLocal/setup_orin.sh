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

ensure_user_owns_path() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Ensuring ownership of: $path"
        sudo chown -R "$USER_NAME:$USER_NAME" "$path" || true
    fi
}

ensure_local_git_excludes() {
    local exclude_file="$PROJECT_DIR/.git/info/exclude"

    if [ ! -d "$PROJECT_DIR/.git" ]; then
        echo "[setup] no .git directory found, skipping local exclude setup"
        return 0
    fi

    mkdir -p "$(dirname "$exclude_file")"
    touch "$exclude_file"

    grep -qxF 'start_aura.sh' "$exclude_file" || echo 'start_aura.sh' >> "$exclude_file"
    grep -qxF 'aura_env/' "$exclude_file" || echo 'aura_env/' >> "$exclude_file"
    grep -qxF 'storage/logs/' "$exclude_file" || echo 'storage/logs/' >> "$exclude_file"
    grep -qxF 'storage/queue/' "$exclude_file" || echo 'storage/queue/' >> "$exclude_file"

    echo "[setup] ensured local git excludes for Jetson-only files"
}

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
  x11-xserver-utils \
  tesseract-ocr

echo "System dependencies installed"

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
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_KV_CACHE_TYPE=q4_0"' "$OLLAMA_SVC"
        CHANGED=1
    fi
    if ! grep -q 'OLLAMA_DRAFT_MODEL' "$OLLAMA_SVC"; then
        sudo sed -i '/\[Service\]/a Environment="OLLAMA_DRAFT_MODEL=llama3.2:1b"' "$OLLAMA_SVC"
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
ollama pull llama3.2:latest
ollama pull llama3.2:1b-instruct-q4_K_M
ollama pull nomic-embed-text

VENV_DIR="$PROJECT_DIR/aura_env"

if [ -d "$VENV_DIR" ]; then
    echo "Existing aura_env detected"
    echo "Reusing existing virtual environment"
    ensure_user_owns_path "$VENV_DIR"
else
    echo "No aura_env found, creating a fresh virtual environment..."
    python3 -m venv "$VENV_DIR"
    ensure_user_owns_path "$VENV_DIR"
fi

ensure_local_git_excludes

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PIP_REQUIRE_VIRTUALENV=true
unset PIP_USER
unset PYTHONUSERBASE

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing Python requirements into existing venv..."
python -m pip install --no-user -r "$PROJECT_DIR/requirements.txt"

echo "Installing Jetson-specific Python libraries..."
python -m pip install --no-user jetson-stats SpeechRecognition Pillow sounddevice

echo "Pre-downloading Whisper models (tiny.en for wake detection, small.en for transcription)..."
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8')" 2>&1 | tail -3 || echo "[WARN] tiny.en download failed — will download on first use"
python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8')" 2>&1 | tail -3 || echo "[WARN] small.en download failed — will download on first use"

echo "Pre-downloading cross-encoder re-ranking model (~65 MB, runs on CPU)..."
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')" 2>&1 | tail -3 || echo "[WARN] cross-encoder download failed — will attempt on first query"

echo "Creating startup launcher..."
cat > "$PROJECT_DIR/start_aura.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$PROJECT_DIR"
VENV_DIR="$VENV_DIR"
DEFAULT_BRANCH="main"

echo "[start_aura] project: \$PROJECT_DIR"
cd "\$PROJECT_DIR"

wait_for_github() {
  local tries=10
  local i
  for ((i=1; i<=tries; i++)); do
    if getent hosts github.com >/dev/null 2>&1; then
      echo "[start_aura] github.com resolved"
      return 0
    fi
    echo "[start_aura] waiting for network... (\$i/\$tries)"
    sleep 3
  done
  return 1
}

repo_has_changes() {
  ! git diff --quiet || \\
  ! git diff --cached --quiet || \\
  [ -n "\$(git ls-files --others --exclude-standard)" ]
}

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CURRENT_BRANCH="\$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "\$DEFAULT_BRANCH")"
  echo "[start_aura] current branch: \$CURRENT_BRANCH"

  if wait_for_github; then
    if repo_has_changes; then
      STASH_MSG="auto-start stash \$(date '+%Y-%m-%d %H:%M:%S')"
      echo "[start_aura] local changes detected, stashing tracked changes..."
      git status --short || true
      if git stash push -m "\$STASH_MSG"; then
        echo "[start_aura] stashed tracked changes as: \$STASH_MSG"
      else
        echo "[start_aura] stash failed, continuing anyway"
      fi
    fi

    echo "[start_aura] fetching latest code..."
    if git fetch origin "\$CURRENT_BRANCH"; then
      echo "[start_aura] pulling latest code..."
      if git pull --ff-only origin "\$CURRENT_BRANCH"; then
        echo "[start_aura] git pull complete"
      else
        echo "[start_aura] git pull failed, continuing with local files"
      fi
    else
      echo "[start_aura] git fetch failed, continuing with local files"
    fi
  else
    echo "[start_aura] network not ready, continuing with local files"
  fi
else
  echo "[start_aura] warning: \$PROJECT_DIR is not a git repo"
fi

source "\$VENV_DIR/bin/activate"
exec python agent/main.py
EOF

chmod +x "$PROJECT_DIR/start_aura.sh"
ensure_user_owns_path "$PROJECT_DIR/start_aura.sh"

echo "Creating udev rule for /dev/ttyACM0..."
sudo tee /etc/udev/rules.d/99-aura-serial.rules > /dev/null <<'EOF'
SUBSYSTEM=="tty", KERNEL=="ttyACM0", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

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
Environment=AURA_LLM_MODEL=llama3.2
Environment=AURA_INTENT_MODEL=llama3.2:1b
Environment=AURA_GRAPH_EXTRACT=false
Environment=AURA_KEEP_ALIVE=2h
Environment=AURA_NUM_CTX=4096
Environment=AURA_MAX_CTX_CHARS=12000
Environment=AURA_TOP_K=8
Environment=AURA_NUM_PREDICT=512
Environment=AURA_NUM_DRAFT=4
Environment=AURA_TEMPERATURE=0.1
Environment=AURA_RERANK_ENABLED=true
Environment=AURA_RERANK_TOP_N=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

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

ensure_user_owns_path "$PROJECT_DIR"
ensure_user_owns_path "$VENV_DIR"

echo ""
echo "===================================="
echo "Setup Complete"
echo "===================================="
echo ""
echo "What this now does:"
echo "1. Reuses your current aura_env"
echo "2. Never deletes aura_env"
echo "3. Updates packages inside the existing venv"
echo "4. Rewrites start_aura.sh"
echo "5. Auto-starts agent/main.py on boot"
echo "6. Stashes tracked git changes on boot before pull"
echo "7. Leaves untracked Jetson-only files alone"
echo "8. Tries git fetch/pull on boot"
echo "9. Auto-applies /dev/ttyACM0 permissions"
echo "10. Auto-rotates display left on login"
echo ""
echo "Useful commands:"
echo "systemctl status $SERVICE_NAME"
echo "journalctl -u $SERVICE_NAME -f"
echo "git -C $PROJECT_DIR stash list"
echo "git -C $PROJECT_DIR check-ignore -v start_aura.sh aura_env storage/logs storage/queue"
echo "cat $PROJECT_DIR/start_aura.sh"
echo ""
echo "No reboot required. This script already restarts the service."
echo ""