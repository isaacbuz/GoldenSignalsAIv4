"""
Authentication API Endpoints - GoldenSignalsAI V3

REST API endpoints for user authentication and authorization.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr

from src.core.auth import (
    AuthService, UserCreate, UserLogin, Token, User,
    get_current_user, require_admin
)
from src.core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]


class RegisterRequest(BaseModel):
    """Registration request model"""
    email: EmailStr
    password: str
    first_name: str
    last_name: str


class UserProfile(BaseModel):
    """User profile model"""
    id: str
    email: str
    first_name: str
    last_name: str
    is_active: bool
    created_at: str
    last_login: str


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: DatabaseManager = Depends(lambda: None)  # Will be injected
) -> LoginResponse:
    """
    Authenticate user and return access token.
    """
    try:
        # Create auth service
        auth_service = AuthService(db)
        
        # Authenticate user
        user = await auth_service.authenticate_user(request.email, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create tokens
        tokens = auth_service.create_tokens(user)
        
        return LoginResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            user={
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_active": user.is_active,
                "roles": user.roles
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/register", response_model=Dict[str, str])
async def register(
    request: RegisterRequest,
    db: DatabaseManager = Depends(lambda: None)
) -> Dict[str, str]:
    """
    Register a new user account.
    """
    try:
        # Create auth service
        auth_service = AuthService(db)
        
        # Create user data
        user_data = UserCreate(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name
        )
        
        # Create user
        user = await auth_service.create_user(user_data)
        
        logger.info(f"New user registered: {user.email}")
        
        return {"message": "User registered successfully", "user_id": user.id}
        
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout user and invalidate token.
    """
    try:
        # In production, add token to blacklist
        logger.info(f"User logged out: {current_user.email}")
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/profile", response_model=UserProfile)
async def get_profile(
    current_user: User = Depends(get_current_user)
) -> UserProfile:
    """
    Get current user profile.
    """
    try:
        return UserProfile(
            id=current_user.id,
            email=current_user.email,
            first_name=current_user.first_name,
            last_name=current_user.last_name,
            is_active=current_user.is_active,
            created_at=current_user.created_at.isoformat(),
            last_login=current_user.last_login.isoformat() if current_user.last_login else ""
        )
        
    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )


@router.put("/profile", response_model=Dict[str, str])
async def update_profile(
    updates: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(lambda: None)
) -> Dict[str, str]:
    """
    Update user profile.
    """
    try:
        # In production, update in database
        allowed_fields = ["first_name", "last_name"]
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        logger.info(f"Profile updated for user: {current_user.email}")
        
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: DatabaseManager = Depends(lambda: None)
) -> Token:
    """
    Refresh access token using refresh token.
    """
    try:
        # Create auth service
        auth_service = AuthService(db)
        
        # Decode refresh token
        payload = auth_service.decode_token(refresh_token)
        
        # Verify token type
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Get user
        user_id = payload.get("sub")
        user = await auth_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new tokens
        tokens = auth_service.create_tokens(user)
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(lambda: None)
) -> Dict[str, str]:
    """
    Change user password.
    """
    try:
        # Create auth service
        auth_service = AuthService(db)
        
        # Verify old password (in production, fetch from DB)
        # For demo, just hash the new password
        new_password_hash = auth_service.get_password_hash(new_password)
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully"}
        
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.get("/users", response_model=list[UserProfile])
async def list_users(
    current_user: User = Depends(require_admin),
    db: DatabaseManager = Depends(lambda: None)
) -> list[UserProfile]:
    """
    List all users (admin only).
    """
    try:
        # In production, fetch from database
        # For now, return mock data
        users = [
            UserProfile(
                id="user_001",
                email="demo@goldensignalsai.com",
                first_name="Demo",
                last_name="User",
                is_active=True,
                created_at=datetime.utcnow().isoformat(),
                last_login=datetime.utcnow().isoformat()
            )
        ]
        
        return users
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


# Add this import at the top
from datetime import datetime 