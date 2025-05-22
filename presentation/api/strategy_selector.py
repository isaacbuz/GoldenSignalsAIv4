from fastapi import APIRouter, Request
from application.services.strategy_selector import StrategySelector

router = APIRouter()
selector = StrategySelector()

@router.post("/strategy_selector/predict")
async def select_strategy(request: Request):
    data = await request.json()
    result = selector.predict(data)
    return {"strategy": result}
