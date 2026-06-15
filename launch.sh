#!/bin/bash

# ─────────────────────────────────────────────────────────────
# BIM AI Engine — Enterprise Launcher
# This script is wrapped inside the .app by Platypus.
# The user never sees this. They just see the app open.
# ─────────────────────────────────────────────────────────────

APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="$HOME/Library/Logs/BIM_AI_Engine.log"
DATA_DIR="$HOME/BIM_AI_Engine_Data"
VENV_DIR="$DATA_DIR/venv"
SRC_DIR="$APP_DIR/src"

# Create user data directory for storing converted files
mkdir -p "$DATA_DIR/output"
mkdir -p "$DATA_DIR/dataset/raw_ifc"
mkdir -p "$DATA_DIR/dataset/dxf_targets"
mkdir -p "$DATA_DIR/models"

echo "BIM AI Engine starting..." | tee "$LOG_FILE"
echo "Data directory: $DATA_DIR" | tee -a "$LOG_FILE"

# ── Check Python ──────────────────────────────────────────────
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display dialog "Python 3 is required to run BIM AI Engine.\n\nPlease install Python from python.org and reopen the app." with title "BIM AI Engine" buttons {"Open python.org", "Cancel"} default button "Open python.org"'
    if [ $? -eq 0 ]; then
        open "https://python.org/downloads"
    fi
    exit 1
fi

echo "Python found: $($PYTHON --version)" | tee -a "$LOG_FILE"

# ── Create virtual environment if first launch ────────────────
if [ ! -d "$VENV_DIR" ]; then
    osascript -e 'display notification "Setting up BIM AI Engine for the first time. This takes 2-3 minutes." with title "BIM AI Engine"'
    echo "First launch — creating virtual environment..." | tee -a "$LOG_FILE"
    "$PYTHON" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    echo "Installing dependencies..." | tee -a "$LOG_FILE"
    pip install --upgrade pip --quiet
    pip install fastapi uvicorn ifcopenshell ezdxf numpy torch torchvision \
                opencv-python-headless pandas streamlit requests supabase \
                python-dotenv --quiet

    echo "Setup complete." | tee -a "$LOG_FILE"
else
    source "$VENV_DIR/bin/activate"
fi

# ── Kill any existing instances ───────────────────────────────
pkill -f "app_orchestrator" 2>/dev/null
pkill -f "streamlit" 2>/dev/null
sleep 1

# ── Start the backend engine ──────────────────────────────────
echo "Starting backend engine..." | tee -a "$LOG_FILE"
cd "$APP_DIR/src"
export DATA_DIR="$DATA_DIR"
"$VENV_DIR/bin/python" app_orchestrator.py >> "$LOG_FILE" 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID" | tee -a "$LOG_FILE"

# Wait for backend to be ready
sleep 3

# ── Start Streamlit UI ────────────────────────────────────────
echo "Starting UI..." | tee -a "$LOG_FILE"
"$VENV_DIR/bin/streamlit" run "$SRC_DIR/app_ui.py" \
    --server.headless true \
    --server.port 8501 \
    --server.maxUploadSize 5000 \
    --browser.gatherUsageStats false >> "$LOG_FILE" 2>&1 &
UI_PID=$!

sleep 2

# ── Open in browser ───────────────────────────────────────────
open "http://localhost:8501"

# ── Keep alive + cleanup on quit ─────────────────────────────
trap "kill $BACKEND_PID $UI_PID 2>/dev/null; exit" INT TERM EXIT
wait $UI_PID