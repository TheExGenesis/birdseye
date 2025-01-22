from supabase import create_client
from typing import Optional, Any
import json
from datetime import datetime


def log_event(
    supabase,
    event_type: str,
    username: Optional[str] = None,
    details: Optional[dict] = None,
):
    """Log an analytics event to Supabase"""
    try:
        supabase.table("analytics").insert(
            {"event_type": event_type, "username": username, "details": details or {}}
        ).execute()
    except Exception as e:
        print(f"Failed to log analytics: {e}")
