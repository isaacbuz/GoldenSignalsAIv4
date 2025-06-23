import os
from fastapi import APIRouter
from typing import Dict

router = APIRouter()

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Golden/ml_models'))

@router.get("/api/model-info")
def get_model_info() -> Dict[str, dict]:
    info = {}
    for fname in ["forecast_model.pkl", "sentiment_model.pkl"]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            stat = os.stat(path)
            info[fname] = {
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime
            }
        else:
            info[fname] = {"error": "File not found"}
    return info
