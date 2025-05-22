# admin_user_management.py
# User and role management endpoints for admin panel (requires Firebase Admin SDK)
from fastapi import APIRouter, Depends, HTTPException, status
from .admin_auth import get_current_user, require_role
from .rate_limit import limiter
import firebase_admin
from firebase_admin import auth as firebase_auth

from infrastructure.auth.jwt_utils import verify_jwt_token
router = APIRouter(prefix="/api/v1/admin/users", tags=["admin-users"])

@router.get("/")
async def list_users(user=Depends(require_role("admin"))):
    # List all users (paginated, 100 max)
    users = []
    page = firebase_auth.list_users()
    for u in page.users:
        users.append({
            "uid": u.uid,
            "email": u.email,
            "displayName": u.display_name,
            "disabled": u.disabled,
            "customClaims": u.custom_claims,
        })
    return users

@router.post("/{uid}/set_role")
@limiter.limit("10/minute")
async def set_user_role(request, uid: str, role: str, user=Depends(require_role("admin"))):
    # Set custom claim for user role
    try:
        firebase_auth.set_custom_user_claims(uid, {"role": role})
        return {"success": True, "message": f"Role for user {uid} set to {role}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk_disable")
@limiter.limit("5/minute")
async def bulk_disable_users(request, uids: list[str], user=Depends(require_role("admin"))):
    results = []
    for uid in uids:
        try:
            firebase_auth.update_user(uid, disabled=True)
            log_admin_action(user, "bulk_disable_user", target=uid)
            results.append({"uid": uid, "success": True})
        except Exception as e:
            log_admin_action(user, "bulk_disable_user", target=uid, outcome="error", details=str(e))
            results.append({"uid": uid, "success": False, "error": str(e)})
    return {"results": results}

@router.post("/{uid}/disable")
@limiter.limit("10/minute")
async def disable_user(request, uid: str, user=Depends(require_role("admin"))):
    try:
        firebase_auth.update_user(uid, disabled=True)
        log_admin_action(user, "disable_user", target=uid)
        return {"success": True, "message": f"User {uid} disabled."}
    except Exception as e:
        log_admin_action(user, "disable_user", target=uid, outcome="error", details=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk_enable")
@limiter.limit("5/minute")
async def bulk_enable_users(request, uids: list[str], user=Depends(require_role("admin"))):
    results = []
    for uid in uids:
        try:
            firebase_auth.update_user(uid, disabled=False)
            log_admin_action(user, "bulk_enable_user", target=uid)
            results.append({"uid": uid, "success": True})
        except Exception as e:
            log_admin_action(user, "bulk_enable_user", target=uid, outcome="error", details=str(e))
            results.append({"uid": uid, "success": False, "error": str(e)})
    return {"results": results}

@router.post("/{uid}/enable")
@limiter.limit("10/minute")
async def enable_user(request, uid: str, user=Depends(require_role("admin"))):
    try:
        firebase_auth.update_user(uid, disabled=False)
        log_admin_action(user, "enable_user", target=uid)
        return {"success": True, "message": f"User {uid} enabled."}
    except Exception as e:
        log_admin_action(user, "enable_user", target=uid, outcome="error", details=str(e))
        raise HTTPException(status_code=500, detail=str(e))

from .rate_limit import limiter

@router.post("/invite")
@limiter.limit("10/minute")
async def invite_user(request, email: str, user=Depends(require_role("admin"))):
    # Create user if not exists
    try:
        existing = None
        try:
            existing = firebase_auth.get_user_by_email(email)
        except firebase_auth.UserNotFoundError:
            pass
        if not existing:
            new_user = firebase_auth.create_user(email=email)
        # Optionally send a custom invite email here
        log_admin_action(user, "invite_user", target=email)
        return {"success": True, "message": f"User {email} invited."}
    except Exception as e:
        log_admin_action(user, "invite_user", target=email, outcome="error", details=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{uid}/reset_password")
@limiter.limit("10/minute")
async def reset_password(request, uid: str, user=Depends(require_role("admin"))):
    try:
        u = firebase_auth.get_user(uid)
        # Send password reset email via Firebase REST API
        FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY")
        if not FIREBASE_WEB_API_KEY:
            raise Exception("FIREBASE_WEB_API_KEY not set in .env")
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_WEB_API_KEY}"
        data = {"requestType": "PASSWORD_RESET", "email": u.email}
        resp = requests.post(url, json=data)
        if resp.status_code != 200:
            raise Exception(f"Failed to send reset email: {resp.text}")
        log_admin_action(user, "reset_password", target=u.email)
        return {"success": True, "message": f"Password reset email sent to {u.email}."}
    except Exception as e:
        log_admin_action(user, "reset_password", target=uid, outcome="error", details=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk_delete")
@limiter.limit("5/minute")
async def bulk_delete_users(request, uids: list[str], user=Depends(require_role("admin"))):
    results = []
    for uid in uids:
        try:
            firebase_auth.delete_user(uid)
            log_admin_action(user, "bulk_delete_user", target=uid)
            results.append({"uid": uid, "success": True})
        except Exception as e:
            log_admin_action(user, "bulk_delete_user", target=uid, outcome="error", details=str(e))
            results.append({"uid": uid, "success": False, "error": str(e)})
    return {"results": results}

@router.post("/{uid}/delete")
@limiter.limit("10/minute")
async def delete_user(request, uid: str, user=Depends(require_role("admin"))):
    try:
        firebase_auth.delete_user(uid)
        log_admin_action(user, "delete_user", target=uid)
        return {"success": True, "message": f"User {uid} deleted."}
    except Exception as e:
        log_admin_action(user, "delete_user", target=uid, outcome="error", details=str(e))
        raise HTTPException(status_code=500, detail=str(e))
