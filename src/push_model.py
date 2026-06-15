"""
push_model.py
-------------
Run this after every training session.
Pushes your new model to Railway automatically.
Friends get smarter output on their next upload.

Usage:
    python src/push_model.py
"""

import subprocess
import shutil
from pathlib import Path

MODEL_LOCAL  = Path(__file__).parent.parent / "models" / "drafting_ai.pth"
PROJECT_ROOT = Path(__file__).parent.parent


def push_model():
    print("=" * 50)
    print("BIM AI — Pushing New Model To Live Server")
    print("=" * 50)

    if not MODEL_LOCAL.exists():
        print("ERROR: No trained model found.")
        print("Train first: python src/annotation_ai.py")
        return

    size_mb = MODEL_LOCAL.stat().st_size / (1024 * 1024)
    print(f"Model found: {size_mb:.2f} MB")
    print("Deploying to Railway...")

    try:
        result = subprocess.run(
            ["railway", "up"],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print("=" * 50)
            print("Deploy complete.")
            print("Live server is now running your new model.")
            print("Friends get smarter output immediately.")
            print("=" * 50)
        else:
            print("Deploy failed. Check terminal output above.")

    except FileNotFoundError:
        print("Railway CLI not found.")
        print("Run: brew install railway")


if __name__ == "__main__":
    push_model()