#!/usr/bin/env python3
"""Fix all FastAPI-related issues in the project."""

import os
import re

def fix_signals_api():
    """Fix the signals API to use proper response models."""

    file_path = 'src/api/v1/signals.py'
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace the problematic endpoint
    content = re.sub(
        r'@router\.get\("/latest", response_model=List\[Signal\]\)',
        '@router.get("/latest", response_model=List[SignalResponse])',
        content
    )

    # Fix the other endpoint that might have similar issues
    content = re.sub(
        r'@router\.get\("/\{symbol\}", response_model=Signal\)',
        '@router.get("/{symbol}/current", response_model=SignalResponse)',
        content
    )

    # Fix the create endpoint
    content = re.sub(
        r'@router\.post\("/", response_model=Signal\)',
        '@router.post("/", response_model=SignalResponse)',
        content
    )

    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed FastAPI response models in {file_path}")

def create_signal_dto():
    """Create proper Signal DTO for API responses."""

    dto_path = 'src/models/dto/signal_dto.py'
    os.makedirs(os.path.dirname(dto_path), exist_ok=True)

    content = '''"""Signal DTOs for API responses."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SignalDTO(BaseModel):
    """Signal data transfer object."""

    signal_id: str
    symbol: str
    signal_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    strength: str
    source: str
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: Optional[float] = None
    reasoning: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    indicators: Optional[Dict[str, float]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SignalsResponse(BaseModel):
    """Response model for multiple signals."""

    signals: List[SignalDTO]
    count: int
    status: str = "success"
'''

    with open(dto_path, 'w') as f:
        f.write(content)
    print(f"✅ Created {dto_path}")

def fix_signal_service():
    """Fix signal service to return proper DTOs."""

    service_path = 'src/services/signal_service.py'
    if os.path.exists(service_path):
        with open(service_path, 'r') as f:
            content = f.read()

        # Add DTO import if not present
        if 'from src.models.dto.signal_dto import SignalDTO' not in content:
            content = 'from src.models.dto.signal_dto import SignalDTO, SignalsResponse\n' + content

        with open(service_path, 'w') as f:
            f.write(content)
        print(f"✅ Fixed imports in {service_path}")

def create_dto_init():
    """Create __init__.py for dto module."""

    init_path = 'src/models/dto/__init__.py'
    content = '''"""Data Transfer Objects."""

from .signal_dto import SignalDTO, SignalsResponse

__all__ = ["SignalDTO", "SignalsResponse"]
'''

    with open(init_path, 'w') as f:
        f.write(content)
    print(f"✅ Created {init_path}")

def main():
    """Run all fixes."""
    print("🚀 Fixing FastAPI issues...\n")

    print("1️⃣ Creating Signal DTOs...")
    create_signal_dto()
    create_dto_init()

    print("\n2️⃣ Fixing signals API...")
    fix_signals_api()

    print("\n3️⃣ Fixing signal service...")
    fix_signal_service()

    print("\n✅ All FastAPI fixes completed!")

    # Test by running a simple test
    print("\nTesting if fixes worked...")
    os.system("python -m pytest tests/test_comprehensive_system.py -v 2>&1 | head -20")

if __name__ == "__main__":
    main()
