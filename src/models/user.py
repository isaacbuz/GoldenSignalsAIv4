"""
User model for authentication and user management
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text
from sqlalchemy.orm import relationship
from .base import BaseModel


class User(BaseModel):
    """User model for authentication"""
    
    __tablename__ = "users"
    
    # Basic user information
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # User profile
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    bio = Column(Text, nullable=True)
    
    # User status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_premium = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    last_login = Column(DateTime(timezone=True), nullable=True)
    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # API settings
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    rate_limit_tier = Column(String(20), default="basic", nullable=False)  # basic, premium, enterprise
    
    # Relationships
    signals = relationship("Signal", back_populates="user", cascade="all, delete-orphan")
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"
    
    @property
    def full_name(self):
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def to_dict(self, include_sensitive=False):
        """Convert to dictionary, optionally including sensitive data"""
        data = super().to_dict()
        
        # Remove sensitive fields by default
        if not include_sensitive:
            data.pop('hashed_password', None)
            data.pop('api_key', None)
        
        # Add computed fields
        data['full_name'] = self.full_name
        
        return data 