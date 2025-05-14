import os
import secrets
import hashlib
from typing import Dict, Any
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from infrastructure.config_manager import config_manager

class SecurityManager:
    """
    Advanced Security Management System
    
    Features:
    - Encryption and decryption
    - Secure token generation
    - Password hashing
    - Multi-factor authentication
    - Audit logging
    """
    
    def __init__(self):
        # Encryption key management
        self._encryption_key = Fernet.generate_key()
        self._cipher_suite = Fernet(self._encryption_key)
        
        # Password hashing
        self._pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT configuration
        self._jwt_secret = config_manager.get('security.jwt_secret', secret=True)
        self._jwt_algorithm = config_manager.get('security.jwt_algorithm', 'HS256')
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data (str): Data to encrypt
        
        Returns:
            str: Encrypted data
        """
        return self._cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data (str): Encrypted data
        
        Returns:
            str: Decrypted data
        """
        return self._cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """
        Securely hash password
        
        Args:
            password (str): Plain text password
        
        Returns:
            str: Hashed password
        """
        return self._pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            plain_password (str): Plain text password
            hashed_password (str): Stored password hash
        
        Returns:
            bool: Password verification result
        """
        return self._pwd_context.verify(plain_password, hashed_password)
    
    def generate_jwt_token(self, user_id: str, expires_delta: timedelta = None) -> str:
        """
        Generate JWT authentication token
        
        Args:
            user_id (str): User identifier
            expires_delta (timedelta, optional): Token expiration
        
        Returns:
            str: JWT token
        """
        if not expires_delta:
            expires_delta = timedelta(minutes=15)
        
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": user_id,
            "exp": expire
        }
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self._jwt_secret, 
            algorithm=self._jwt_algorithm
        )
        
        return encoded_jwt
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token
        
        Args:
            token (str): JWT token
        
        Returns:
            Dict[str, Any]: Decoded token payload
        """
        try:
            payload = jwt.decode(
                token, 
                self._jwt_secret, 
                algorithms=[self._jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def generate_mfa_token(self) -> str:
        """
        Generate multi-factor authentication token
        
        Returns:
            str: MFA token
        """
        return secrets.token_urlsafe(32)
    
    def audit_log(self, event: str, user_id: str = None, details: Dict[str, Any] = None):
        """
        Create secure audit log entry
        
        Args:
            event (str): Event description
            user_id (str, optional): User identifier
            details (Dict[str, Any], optional): Additional event details
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "user_id": user_id,
            "details": details or {}
        }
        
        # Hash sensitive information
        log_entry_hash = hashlib.sha256(
            str(log_entry).encode()
        ).hexdigest()
        
        # TODO: Implement secure log storage (e.g., encrypted database)
        print(f"AUDIT: {log_entry_hash}")

# Singleton instance
security_manager = SecurityManager()
