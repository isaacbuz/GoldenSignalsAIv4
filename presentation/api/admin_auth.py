# admin_auth.py
# FastAPI dependency for Firebase Admin authentication and RBAC
import os
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import sys

# Patch firebase_admin for test mode if needed
if os.getenv("TEST_MODE") == "1":
    from unittest import mock
    sys.modules['firebase_admin'] = mock.MagicMock()
    sys.modules['firebase_admin.auth'] = mock.MagicMock()
    sys.modules['firebase_admin.credentials'] = mock.MagicMock()

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from typing import Optional

# Initialize Firebase Admin SDK (singleton)
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIREBASE_ADMIN_CREDENTIALS", "firebase-adminsdk.json"))
    firebase_admin.initialize_app(cred)

bearer_scheme = HTTPBearer()

# Example role mapping (replace with DB or custom claims in production)
USER_ROLES = {
    # 'user_uid': 'admin',
}

async def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        # Get role from custom claims or mapping
        role = decoded_token.get("role") or USER_ROLES.get(uid, "user")
        return {"uid": uid, "email": email, "role": role}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase authentication token",
        )

def require_role(required_role: str):
    async def role_dependency(user = Depends(get_current_user)):
        if user["role"] != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges"
            )
        return user
    return role_dependency
