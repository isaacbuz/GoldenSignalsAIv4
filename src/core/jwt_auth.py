"""
JWT Authentication implementation for GoldenSignalsAI.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
import redis
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from src.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# JWT settings
SECRET_KEY = settings.JWT_SECRET_KEY or "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: list[str] = []

class JWTAuth:
    """JWT Authentication handler."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def create_access_token(
        self, 
        data: dict, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(
        self, 
        data: dict, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Store refresh token in Redis if available
        if self.redis_client:
            user_id = data.get("user_id")
            if user_id:
                self.redis_client.setex(
                    f"refresh_token:{user_id}:{encoded_jwt}",
                    timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
                    json.dumps({"created_at": datetime.utcnow().isoformat()})
                )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode a JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Verify token type
            if payload.get("type") != token_type:
                raise credentials_exception
            
            # Check if token is revoked (for refresh tokens)
            if token_type == "refresh" and self.redis_client:
                user_id = payload.get("user_id")
                if user_id and not self.redis_client.exists(f"refresh_token:{user_id}:{token}"):
                    raise credentials_exception
            
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            scopes: list = payload.get("scopes", [])
            
            if username is None:
                raise credentials_exception
            
            token_data = TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise credentials_exception
        
        return token_data
    
    def revoke_token(self, token: str, user_id: int) -> bool:
        """Revoke a refresh token."""
        if self.redis_client:
            return self.redis_client.delete(f"refresh_token:{user_id}:{token}") > 0
        return False
    
    def revoke_all_user_tokens(self, user_id: int) -> int:
        """Revoke all refresh tokens for a user."""
        if self.redis_client:
            pattern = f"refresh_token:{user_id}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
        return 0

# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """Get current user from JWT token."""
    auth = JWTAuth()
    return auth.verify_token(token)

async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Get current active user."""
    # Here you would typically check if user is active in database
    return current_user

def require_scopes(*required_scopes: str):
    """Decorator to require specific scopes."""
    async def scope_checker(
        current_user: TokenData = Depends(get_current_active_user)
    ) -> TokenData:
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions. Required scope: {scope}"
                )
        return current_user
    return scope_checker 