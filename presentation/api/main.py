# presentation/api/main.py
# Purpose: Implements the FastAPI backend for GoldenSignalsAI, providing endpoints for
# predictions, dashboard data, user preferences, and health checks. Includes JWT authentication
# for secure access, optimized for options trading workflows.

import logging
import os
from datetime import datetime, timedelta
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="GoldenSignalsAI API")

# Security setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock user database
users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("password1"),
        "disabled": False,
    }
}


class Token:
    access_token: str
    token_type: str


class User:
    username: str
    disabled: bool = False


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)
    return None


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username)
    if user is None:
        raise credentials_exception
    return user


@app.post("/token", response_model=Dict)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info({"message": f"User {user.username} logged in"})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/health")
async def health_check():
    logger.info({"message": "Health check endpoint called"})
    return {"status": "healthy"}


@app.post("/predict")
async def predict(data: Dict, current_user: User = Depends(get_current_user)):
    logger.info({"message": f"Prediction requested by {current_user.username}"})
    # Mock prediction response
    return {"status": "Prediction successful", "symbol": data.get("symbol", "AAPL")}


@app.get("/dashboard/{symbol}")
async def get_dashboard_data(
    symbol: str, current_user: User = Depends(get_current_user)
):
    logger.info(
        {"message": f"Dashboard data requested for {symbol} by {current_user.username}"}
    )
    # Mock dashboard data
    return {"symbol": symbol, "price": 150.0, "trend": "up"}
