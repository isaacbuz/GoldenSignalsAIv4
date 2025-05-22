from fastapi import APIRouter, Request
from application.services.forecasting_agent import ForecastingAgent
import pandas as pd

router = APIRouter()
forecast_agent = ForecastingAgent()

@router.post("/forecast/train")
async def forecast_train(request: Request):
    data = await request.json()
    X = pd.DataFrame(data["X"])
    y = pd.Series(data["y"])
    forecast_agent.train(X, y)
    return {"status": "trained"}

@router.post("/forecast/predict")
async def forecast_predict(request: Request):
    data = await request.json()
    X = pd.DataFrame(data["X"])
    preds = forecast_agent.predict(X)
    return {"predictions": preds.tolist()}
