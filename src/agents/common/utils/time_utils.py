from datetime import datetime, timezone

def get_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc) 