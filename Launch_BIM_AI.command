#!/bin/bash
cd "$(dirname "$0")"
echo "Initializing Enterprise BIM AI Blueprint Engine..."
source venv/bin/activate
streamlit run src/app_ui.py --server.headless true