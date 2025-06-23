"""
Authentication Service for GoldenSignalsAI V3
Handles JWT tokens, user authentication, and authorization
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from src.config.settings import settings
from src.core.database import DatabaseManager

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# HTTP Bearer scheme
security = HTTPBearer(auto_error=False)


class UserCreate(BaseModel):
    """User registration model"""
    email: EmailStr
    password: str
    first_name: str
    last_name: str


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    email: str
    roles: list[str] = ["user"]


class User(BaseModel):
    """User model"""
    id: str
    email: str
    first_name: str
    last_name: str
    is_active: bool = True
    roles: list[str] = ["user"]
    created_at: datetime
    last_login: Optional[datetime] = None


class AuthService:
    """
    Authentication service handling user authentication and authorization
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.security.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.security.secret_key,
            algorithm=settings.security.algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        
        expire = datetime.utcnow() + timedelta(days=7)  # Refresh tokens last 7 days
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.security.secret_key,
            algorithm=settings.security.algorithm
        )
        
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.security.secret_key,
                algorithms=[settings.security.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        # In production, fetch from database
        # For now, using mock data
        if email == "demo@goldensignalsai.com" and password == "demo123":
            return User(
                id="user_001",
                email=email,
                first_name="Demo",
                last_name="User",
                is_active=True,
                roles=["user", "trader"],
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID from database"""
        # In production, fetch from database
        # For now, using mock data
        if user_id == "user_001":
            return User(
                id=user_id,
                email="demo@goldensignalsai.com",
                first_name="Demo",
                last_name="User",
                is_active=True,
                roles=["user", "trader"],
                created_at=datetime.utcnow()
            )
        return None
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create new user"""
        # In production, save to database
        # For now, return mock user
        hashed_password = self.get_password_hash(user_data.password)
        
        return User(
            id="user_002",
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            is_active=True,
            roles=["user"],
            created_at=datetime.utcnow()
        )
    
    def create_tokens(self, user: User) -> Token:
        """Create access and refresh tokens for user"""
        token_data = {
            "sub": user.id,
            "email": user.email,
            "roles": user.roles
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.security.access_token_expire_minutes * 60
        )


# Dependency functions
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: DatabaseManager = Depends(lambda: None)  # Will be injected
) -> User:
    """
    Get current authenticated user from JWT token
    """
    if not credentials:
        # Allow unauthenticated access in debug mode
        if settings.debug:
            return User(
                id="dev_user",
                email="dev@goldensignalsai.com",
                first_name="Dev",
                last_name="User",
                is_active=True,
                roles=["user", "admin"],
                created_at=datetime.utcnow()
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create auth service
    auth_service = AuthService(db)
    
    # Decode token
    payload = auth_service.decode_token(credentials.credentials)
    
    # Verify token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Get user
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user = await auth_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_user_ws(token: str) -> Optional[User]:
    """
    Get current user for WebSocket connections
    """
    try:
        # Create mock auth service
        auth_service = AuthService(None)
        
        # Decode token
        payload = auth_service.decode_token(token)
        
        # Get user
        user_id = payload.get("sub")
        if user_id:
            return await auth_service.get_user_by_id(user_id)
    except:
        pass
    
    return None


def require_roles(required_roles: list[str]):
    """
    Dependency to require specific roles
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker


# Convenience dependencies
require_admin = require_roles(["admin"])
require_trader = require_roles(["trader", "admin"]) 