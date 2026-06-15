#!/bin/bash
# ─────────────────────────────────────────────────────────────
# BIM AI Engine — Enterprise Mac Bootstrapper
# ─────────────────────────────────────────────────────────────

# Fix: Set all runtime variables to target user home directory
DATA_DIR="$HOME/BIM_AI_Engine_Data"
VENV_DIR="$DATA_DIR/venv"
LOG_FILE="$DATA_DIR/engine.log"

mkdir -p "$DATA_DIR/output"
mkdir -p "$DATA_DIR/dataset/raw_ifc"
mkdir -p "$DATA_DIR/models"

# Use Platypus's internal extraction path to locate your bundled source files
APP_DIR="$1/Contents/Resources"
if [ ! -d "$APP_DIR/src" ]; then
    APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
SRC_DIR="$APP_DIR/src"

echo "Initializing Engine..." > "$LOG_FILE"

# Find local system python installation
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display dialog "Python 3 is required.\nInstall from python.org." with title "BIM AI Engine"'
    exit 1
fi

# Build or activate local user runtime environment
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    pip install fastapi uvicorn ifcopenshell ezdxf numpy torch streamlit requests supabase python-dotenv --quiet
else
    source "$VENV_DIR/bin/activate"
fi

# Boot up the pipeline components
cd "$SRC_DIR"
export DATA_DIR="$DATA_DIR"
"$VENV_DIR/bin/python" app_orchestrator.py >> "$LOG_FILE" 2>&1 &
BACKEND_PID=$!

sleep 2
"$VENV_DIR/bin/streamlit" run app_ui.py --server.headless true --server.port 8501 >> "$LOG_FILE" 2>&1 &
UI_PID=$!

sleep 1
open "http://localhost:8501"

trap "kill $BACKEND_PID $UI_PID 2>/dev/null; exit" INT TERM EXIT
wait $UI_PID