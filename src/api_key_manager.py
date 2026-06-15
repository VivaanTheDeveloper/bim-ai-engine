"""
api_key_manager.py
------------------
Manages enterprise API keys for firm authentication.
Every firm gets a unique key. You control who gets access.
"""

import secrets
import string
from supabase import create_client
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def generate_api_key(firm_name: str, email: str) -> str:
    """
    Generate a new enterprise API key for a firm.
    Run this manually when you onboard a new client.

    Usage:
        python src/api_key_manager.py --firm "Hafeez Contractor" --email "contact@hafeez.com"
    """
    client = get_client()

    # Generate cryptographically secure key
    alphabet = string.ascii_letters + string.digits
    raw_key = ''.join(secrets.choice(alphabet) for _ in range(48))
    api_key = f"bim_{raw_key}"

    # Store in Supabase
    client.table("api_keys").insert({
        "firm_name": firm_name,
        "email": email,
        "api_key": api_key,
        "plan": "enterprise",
        "is_active": True
    }).execute()

    print("=" * 60)
    print(f"API Key Generated for: {firm_name}")
    print(f"Email: {email}")
    print(f"API Key: {api_key}")
    print("=" * 60)
    print("Send this key to the firm. They enter it in their app.")
    print("They can now process unlimited IFC files.")

    return api_key


def validate_api_key(api_key: str) -> dict:
    """
    Validates an API key on every request.
    Returns firm info if valid, None if invalid.
    """
    if not api_key:
        return None

    client = get_client()

    result = client.table("api_keys")\
        .select("*")\
        .eq("api_key", api_key)\
        .eq("is_active", True)\
        .execute()

    if not result.data:
        return None

    return result.data[0]


def increment_usage(api_key: str):
    """Tracks how many files each firm has processed."""
    client = get_client()
    client.rpc("increment_requests", {"key": api_key}).execute()


def log_job(api_key: str, firm_name: str, filename: str,
            wall_count: int, dxf_url: str, csv_url: str):
    """Logs every processing job for analytics."""
    client = get_client()
    client.table("processing_jobs").insert({
        "api_key": api_key,
        "firm_name": firm_name,
        "filename": filename,
        "wall_count": wall_count,
        "status": "completed",
        "dxf_url": dxf_url,
        "csv_url": csv_url
    }).execute()


def deactivate_key(api_key: str):
    """Deactivate a firm's access instantly."""
    client = get_client()
    client.table("api_keys")\
        .update({"is_active": False})\
        .eq("api_key", api_key)\
        .execute()
    print(f"Key deactivated: {api_key}")


def list_all_firms():
    """See all your clients and their usage."""
    client = get_client()
    result = client.table("api_keys").select("*").execute()

    print("\n" + "=" * 60)
    print("ALL ENTERPRISE CLIENTS")
    print("=" * 60)
    for firm in result.data:
        status = "ACTIVE" if firm["is_active"] else "INACTIVE"
        print(f"\nFirm:     {firm['firm_name']}")
        print(f"Email:    {firm['email']}")
        print(f"Jobs:     {firm['requests_total']}")
        print(f"Status:   {status}")
        print(f"Joined:   {firm['created_at'][:10]}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if "--firm" in sys.argv and "--email" in sys.argv:
        firm_idx  = sys.argv.index("--firm") + 1
        email_idx = sys.argv.index("--email") + 1
        generate_api_key(sys.argv[firm_idx], sys.argv[email_idx])

    elif "--list" in sys.argv:
        list_all_firms()

    elif "--deactivate" in sys.argv:
        key_idx = sys.argv.index("--deactivate") + 1
        deactivate_key(sys.argv[key_idx])

    else:
        print("Usage:")
        print("  Generate key: python src/api_key_manager.py --firm 'Firm Name' --email 'email@firm.com'")
        print("  List firms:   python src/api_key_manager.py --list")
        print("  Deactivate:   python src/api_key_manager.py --deactivate bim_xxxxx")