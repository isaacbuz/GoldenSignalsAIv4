from fastapi import Header, HTTPException, Depends
from firebase_admin import auth as firebase_auth

# Simulate an API key database (in production, use Redis or Firebase claims)
AUTHORIZED_KEYS = {
    "demo-api-key-123": {"user_id": "demo-user", "permissions": ["read", "write"]}
}

def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key not in AUTHORIZED_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return AUTHORIZED_KEYS[x_api_key]
