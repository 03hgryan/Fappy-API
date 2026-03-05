import os
from datetime import datetime, timezone
from supabase import create_client, Client

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        _client = create_client(url, key)
    return _client


def insert_feedback(message: str, email: str | None = None, user_id: str | None = None) -> dict:
    sb = get_supabase()
    row: dict = {
        "message": message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if email:
        row["email"] = email
    if user_id:
        row["user_id"] = user_id
    result = sb.table("feedback").insert(row).execute()
    return result.data[0] if result.data else row


def upsert_user(google_id: str, email: str, name: str | None, picture_url: str | None) -> dict:
    sb = get_supabase()
    row = {
        "google_id": google_id,
        "email": email,
        "name": name,
        "picture_url": picture_url,
        "last_login": datetime.now(timezone.utc).isoformat(),
    }
    result = sb.table("users").upsert(row, on_conflict="google_id").execute()
    return result.data[0] if result.data else row
