import os
from fastapi import APIRouter, HTTPException
import yaml
from typing import Dict

router = APIRouter()

AGENT_WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../Golden/config/agent_weights.yaml')
)

def load_agent_weights() -> Dict[str, float]:
    with open(AGENT_WEIGHTS_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_agent_weights(weights: Dict[str, float]):
    with open(AGENT_WEIGHTS_PATH, 'w') as f:
        yaml.dump(weights, f)

@router.get("/api/agent-weights")
def get_agent_weights():
    try:
        weights = load_agent_weights()
        return {"weights": weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/agent-weights")
def update_agent_weights(weights: Dict[str, float]):
    try:
        save_agent_weights(weights)
        return {"message": "Agent weights updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
