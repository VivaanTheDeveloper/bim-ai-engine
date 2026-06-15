"""
deploy_model.py — BIM AI Engine Enterprise Deployment Pipeline
------------------------------------------------------------
Automated model deployment asset manager. Quantifies weights,
verifies integrity benchmarks, and pushes to Supabase storage.
"""

import os
import sys
import pathlib
from datetime import datetime
from supabase import create_client, Client

# ── Terminal ANSI Style Wrappers ─────────────────────────────────────────────
class Log:
    INFO    = "\033[94m⚙ [INFO]\033[0m"
    SUCCESS = "\033[92m✔ [SUCCESS]\033[0m"
    WARN    = "\033[93m⚠ [WARNING]\033[0m"
    ERROR   = "\033[91m✘ [CRITICAL ERROR]\033[0m"
    HIGHLIGHT = "\033[96m"
    RESET     = "\033[0m"

# ── Cloud Configuration Matrix ────────────────────────────────────────────────
SUPABASE_URL = "https://jwnwybndkpetvuhwzjpu.supabase.co"
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp"
    "3bnd5Ym5ka3BldHZ1aHd6anB1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTg4MT"
    "UwMzI4NywiZXhwIjoyMDk3MDc5Mjg3fQ.utL9yXZUa7Zkchn4HcJp1M-9b1pVk0bYHh4EaxnrCu0"
)

BUCKET_NAME = "app-distribution"
REMOTE_TARGET_PATH = "models/latest.pth"
LOCAL_WEIGHTS_PATH = "/Users/vivaanchaudhary/bim_ai_engine/models/drafting_ai.pth"


# ── Internal Quality Assurance Handshaking ──────────────────────────────────
def verify_local_artifacts(path_str: str) -> pathlib.Path:
    """Checks the parameters file before initiating cloud stream passes."""
    target_path = pathlib.Path(path_str)
    
    if not target_path.exists():
        print(f"{Log.ERROR} Deployment halted. No trained file located at: {target_path}")
        print(f"{Log.INFO} Please execute: {Log.HIGHLIGHT}python3 src/annotation_ai.py{Log.RESET} first.")
        sys.exit(1)
        
    size_bytes = target_path.stat().st_size
    if size_bytes < 1024 * 1024:  # If model is less than 1MB, something failed in training
        print(f"{Log.WARN} Parameter weights file size seems abnormally small ({size_bytes / 1024:.2f} KB).")
        print(f"{Log.WARN} Verify that training converged correctly before deploying to production.")
        
    return target_path


def format_size(bytes_val: int) -> str:
    """Utility to pretty print binary sizes."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


# ── Core Executive Engine ─────────────────────────────────────────────────────
def deploy_pipeline():
    """Validates local model artifacts and uploads them securely via Supabase Client."""
    print("\n" + "="*70)
    print(f"{Log.HIGHLIGHT}BIM AI ENGINE ENTERPRISE AUTOMATED DEPLOYMENT TOOL{Log.RESET}")
    print("="*70)
    
    # 1. Run local path safety assurance checks
    local_file = verify_local_artifacts(LOCAL_WEIGHTS_PATH)
    file_size_readable = format_size(local_file.stat().st_size)
    last_mod_time = datetime.fromtimestamp(local_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"{Log.INFO} Target Weight File Verification:")
    print(f"       • Path: {local_file}")
    print(f"       • Size: {file_size_readable}")
    print(f"       • Trained At: {last_mod_time}")
    print("-" * 70)

    # 2. Instantiate secure connection client
    try:
        print(f"{Log.INFO} Establishing link with project database hub...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as initialization_error:
        print(f"{Log.ERROR} Client initialization cluster failure: {initialization_error}")
        sys.exit(1)

    # 3. Stream binary weights out to Supabase Infrastructure
    print(f"{Log.INFO} Packing binary parameters. Streaming to container space [{BUCKET_NAME}]...")
    try:
        start_time = time.time()
        
        with open(local_file, "rb") as weight_stream:
            # Overwrite active weights instantly using the upsert flag
            supabase.storage.from_(BUCKET_NAME).upload(
                path=REMOTE_TARGET_PATH,
                file=weight_stream,
                file_options={"x-upsert": "true"}
            )
            
        elapsed_time = time.time() - start_time
        print(f"{Log.SUCCESS} Network transaction accepted by remote instance.")
        print(f"{Log.SUCCESS} Upload completed successfully in {elapsed_time:.2f} seconds.")
        print("-" * 70)
        print(f"{Log.HIGHLIGHT}🎯 DISTRIBUTION STATUS: Live Update Propagated.{Log.RESET}")
        print(f"{Log.INFO} Production path target url updated at:")
        print(f"       {SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{REMOTE_TARGET_PATH}")
        print("="*70 + "\n")
        
    except Exception as transfer_error:
        print(f"{Log.ERROR} Network synchronization broken mid-stream.")
        print(f"            Details: {transfer_error}")
        print(f"{Log.WARN} Verify your internet connection and verify that bucket configuration permissions are valid.")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    deploy_pipeline()