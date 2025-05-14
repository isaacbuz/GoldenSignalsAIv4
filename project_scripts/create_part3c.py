# create_part3c.py
# Purpose: Creates files in the presentation/ directory for the GoldenSignalsAI project,
# including the FastAPI backend, React frontend, and end-to-end tests. Incorporates improvements
# like secure API endpoints and a user-friendly dashboard for options trading.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part3c():
    """Create files in presentation/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating presentation files in {base_dir}"})

    # Define presentation directory files
    presentation_files = {
        "presentation/__init__.py": """# presentation/__init__.py
# Purpose: Marks the presentation directory as a Python package, enabling imports
# for API, frontend, and test components.
""",
        "presentation/api/main.py": """# presentation/api/main.py
# Purpose: Implements the FastAPI backend for GoldenSignalsAI, providing endpoints for
# predictions, dashboard data, user preferences, and health checks. Includes JWT authentication
# for secure access, optimized for options trading workflows.

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import logging
from typing import Dict
import os

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
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
        "disabled": False
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
async def get_dashboard_data(symbol: str, current_user: User = Depends(get_current_user)):
    logger.info({"message": f"Dashboard data requested for {symbol} by {current_user.username}"})
    # Mock dashboard data
    return {"symbol": symbol, "price": 150.0, "trend": "up"}
""",
        "presentation/frontend/src/App.js": """// presentation/frontend/src/App.js
// Purpose: Implements the React frontend for GoldenSignalsAI, providing a dashboard for
// viewing trading signals, agent activity, and market data, with options trading-specific features.

import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [symbol, setSymbol] = useState('AAPL');
  const [data, setData] = useState(null);
  const [token, setToken] = useState(null);

  useEffect(() => {
    // Login to get token
    fetch('http://localhost:8000/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        'username': 'user1',
        'password': 'password1'
      })
    })
      .then(response => response.json())
      .then(data => setToken(data.access_token))
      .catch(error => console.error('Login failed:', error));
  }, []);

  useEffect(() => {
    if (token) {
      fetch(`http://localhost:8000/dashboard/${symbol}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => response.json())
        .then(data => setData(data))
        .catch(error => console.error('Error fetching dashboard data:', error));
    }
  }, [symbol, token]);

  return (
    <div className="App">
      <h1>GoldenSignalsAI Dashboard</h1>
      <select onChange={(e) => setSymbol(e.target.value)} value={symbol}>
        <option value="AAPL">AAPL</option>
        <option value="GOOGL">GOOGL</option>
        <option value="MSFT">MSFT</option>
      </select>
      {data ? (
        <div>
          <h2>{data.symbol}</h2>
          <p>Price: ${data.price}</p>
          <p>Trend: {data.trend}</p>
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default App;
""",
        "presentation/frontend/src/App.css": """/* presentation/frontend/src/App.css */
/* Purpose: Styles the React frontend for GoldenSignalsAI. */

.App {
  text-align: center;
  padding: 20px;
}

h1 {
  color: #333;
}

select {
  margin: 20px;
  padding: 5px;
}

div {
  margin-top: 20px;
}
""",
        "presentation/frontend/package.json": """{
  "name": "goldensignalsai-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
""",
        "presentation/tests/test_api.py": """# presentation/tests/test_api.py
# Purpose: End-to-end tests for the FastAPI backend of GoldenSignalsAI, ensuring API
# endpoints work as expected for options trading workflows.

import pytest
import httpx
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_check():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_login():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/token", data={"username": "user1", "password": "password1"})
        assert response.status_code == 200
        assert "access_token" in response.json()
""",
    }

    # Write presentation directory files
    for file_path, content in presentation_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 3c: presentation/ created successfully")


if __name__ == "__main__":
    create_part3c()
