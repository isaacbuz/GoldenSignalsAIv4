from fastapi import APIRouter
import subprocess

router = APIRouter()

@router.post("/api/train")
def train_models():
    result = subprocess.run(["python", "backend/ml_training/retrain_all.py"], capture_output=True, text=True)
    return {"status": "success", "output": result.stdout}

