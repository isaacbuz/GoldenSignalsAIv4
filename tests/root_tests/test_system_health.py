#!/usr/bin/env python3
"""
GoldenSignalsAI V3 - System Health Check
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def check_database():
    """Test database connectivity"""
    try:
        from core.database import DatabaseManager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        health_ok = await db_manager.health_check()
        await db_manager.close()
        return "✅ WORKING" if health_ok else "❌ FAILED"
    except Exception as e:
        return f"❌ FAILED: {str(e)}"

def check_ml_models():
    """Check ML models"""
    try:
        models_path = Path("ml_models")
        forecast_model = models_path / "forecast_model.pkl"

        if forecast_model.exists():
            with open(forecast_model, 'r') as f:
                content = f.read().strip()
                if content.startswith("# Placeholder"):
                    return "❌ FAILED: Models are placeholders"
        return "✅ WORKING"
    except Exception as e:
        return f"❌ FAILED: {str(e)}"

async def main():
    print("🔍 GOLDENSIGNALS AI V3 - SYSTEM HEALTH CHECK")
    print("=" * 50)

    # Check ML Models
    ml_status = check_ml_models()
    print(f"ML Models:     {ml_status}")

    # Check Database
    db_status = await check_database()
    print(f"Database:      {db_status}")

    # Check imports
    try:
        from main import app
        print(f"FastAPI:       ✅ WORKING")
    except Exception as e:
        print(f"FastAPI:       ❌ FAILED: {str(e)}")

    print("\n🎯 CRITICAL FIXES NEEDED:")
    print("1. Fix broken imports: python fix_imports.py")
    print("2. Train ML models: cd ml_training && python train_models.py")
    print("3. Test database: python test_database.py")

if __name__ == "__main__":
    asyncio.run(main())
