from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from infrastructure.auth.jwt_utils import verify_jwt_token

router = APIRouter(prefix="/api/v1/backtest_playground", tags=["backtest-playground"])

class StrategyConfig(BaseModel):
    logic: str
    params: dict

@router.post("/run_strategy")
async def run_strategy(config: StrategyConfig, user=Depends(verify_jwt_token)):
    # TODO: Run strategy backtest
    # Placeholder response
    return {"result": "success", "pnl": [100, 102, 105, 103]}

class PortfolioConfig(BaseModel):
    allocations: dict
    symbols: list

@router.post("/portfolio_sim")
async def portfolio_sim(config: PortfolioConfig, user=Depends(verify_jwt_token)):
    # TODO: Run portfolio simulation
    # Placeholder response
    return {"result": "success", "history": [10000, 10100, 10250, 10400]}
