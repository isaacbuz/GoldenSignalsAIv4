from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict
from infrastructure.auth.jwt_utils import verify_jwt_token

router = APIRouter(prefix="/api/v1/signal_explain", tags=["signal-explain"])

# Mock model and data (replace with real model instance and features)
model = RandomForestClassifier(n_estimators=100)
X_sample = np.random.rand(10, 5)  # 10 samples, 5 features
model.fit(X_sample, np.random.randint(0, 2, 10))
explainer = shap.TreeExplainer(model)

class ExplainRequest(BaseModel):
    symbol: str
    model: str = "xgboost"
    features: List[float]

class ExplainResponse(BaseModel):
    feature_importance: List[Dict[str, float]]

@router.post("/feature_importance")
async def feature_importance(request: ExplainRequest, user=Depends(verify_jwt_token)):
    try:
        shap_values = explainer.shap_values(np.array([request.features]))
        feature_names = [f"feature_{i}" for i in range(len(request.features))]
        shap_scores = shap_values[1][0]  # class 1 shap values
        importance = [{"feature": name, "weight": float(score)} for name, score in zip(feature_names, shap_scores)]
        importance.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
