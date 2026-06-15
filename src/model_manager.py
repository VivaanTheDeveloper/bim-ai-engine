"""
model_manager.py
----------------
Handles model versioning and distribution.

When you train a new model on your laptop:
    python src/model_manager.py --push --version "v2" --notes "Trained on 50 hospital pairs"

Every firm's app pulls the latest model automatically
on their next processing request.
"""

import os
import json
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

SUPABASE_URL        = os.getenv("SUPABASE_URL")
SUPABASE_KEY        = os.getenv("SUPABASE_SERVICE_KEY")
MODEL_LOCAL         = Path(__file__).parent.parent / "models" / "drafting_ai.pth"
MODEL_CACHE         = Path(__file__).parent.parent / "models" / "drafting_ai.pth"
VERSION_CACHE_FILE  = Path(__file__).parent.parent / "models" / "current_version.json"


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def push_model(version: str, description: str = "", accuracy_notes: str = ""):
    """
    Push your newly trained model to Supabase Storage.
    Run this after every training session on your laptop.
    """
    if not MODEL_LOCAL.exists():
        print("ERROR: No trained model found at models/drafting_ai.pth")
        print("Train first: python3 src/annotation_ai.py")
        return

    client   = get_client()
    size_mb  = MODEL_LOCAL.stat().st_size / (1024 * 1024)

    print(f"Pushing model {version} ({size_mb:.1f} MB) to cloud...")

    # Upload model file to Supabase Storage
    with open(MODEL_LOCAL, "rb") as f:
        model_bytes = f.read()

    storage_path = f"models/drafting_ai_{version}.pth"

    client.storage.from_("bim-outputs").upload(
        path=storage_path,
        file=model_bytes,
        file_options={"content-type": "application/octet-stream"}
    )

    # Also upload as "latest" so servers always know what to pull
    try:
        client.storage.from_("bim-outputs").remove(["models/latest.pth"])
    except Exception:
        pass

    client.storage.from_("bim-outputs").upload(
        path="models/latest.pth",
        file=model_bytes,
        file_options={"content-type": "application/octet-stream"}
    )

    # Deactivate previous versions in DB
    client.table("model_versions")\
        .update({"is_active": False})\
        .eq("is_active", True)\
        .execute()

    # Log new version
    client.table("model_versions").insert({
        "version": version,
        "description": description,
        "accuracy_notes": accuracy_notes,
        "is_active": True
    }).execute()

    print("=" * 60)
    print(f"Model {version} pushed successfully.")
    print("Every client gets this model on their next request.")
    print("Zero restarts required. Zero code changes.")
    print("=" * 60)


def pull_latest_model() -> bool:
    """
    Called automatically by inference_engine.py on the server.
    Checks if a newer model exists in cloud and downloads it.
    """
    try:
        client = get_client()

        # Check what version is active in cloud
        result = client.table("model_versions")\
            .select("*")\
            .eq("is_active", True)\
            .execute()

        if not result.data:
            print("No model in cloud yet. Using geometric fallback.")
            return False

        cloud_version = result.data[0]["version"]

        # Check what version we have locally cached
        local_version = None
        if VERSION_CACHE_FILE.exists():
            with open(VERSION_CACHE_FILE) as f:
                local_version = json.load(f).get("version")

        if local_version == cloud_version and MODEL_CACHE.exists():
            print(f"Model {cloud_version} already cached. No download needed.")
            return True

        # Download latest model from Supabase Storage
        print(f"Downloading model {cloud_version} from cloud...")
        MODEL_CACHE.parent.mkdir(exist_ok=True)

        response = client.storage.from_("bim-outputs")\
            .download("models/latest.pth")

        with open(MODEL_CACHE, "wb") as f:
            f.write(response)

        # Cache the version number
        with open(VERSION_CACHE_FILE, "w") as f:
            json.dump({"version": cloud_version, "pulled_at": str(datetime.now())}, f)

        print(f"Model {cloud_version} ready. Smarter output starts now.")
        return True

    except Exception as e:
        print(f"Could not pull model from cloud: {e}")
        print("Using geometric fallback.")
        return False


if __name__ == "__main__":
    import sys

    if "--push" in sys.argv:
        version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else f"v{datetime.now().strftime('%Y%m%d')}"
        notes   = sys.argv[sys.argv.index("--notes") + 1] if "--notes" in sys.argv else ""
        push_model(version=version, accuracy_notes=notes)
    else:
        print("Usage: python src/model_manager.py --push --version 'v2' --notes 'Trained on 50 pairs'")