from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict
from infrastructure.auth.jwt_utils import verify_jwt_token
from application.services.watchlist_manager import WatchlistManager

router = APIRouter(prefix="/api/v1/watchlist", tags=["watchlist"])

watchlist_manager = WatchlistManager()

class WatchlistRequest(BaseModel):
    user_id: str
    ticker: str
    tags: List[str]

class WatchlistResponse(BaseModel):
    watchlist: List[Dict]

@router.post("/add", response_model=WatchlistResponse)
async def add_to_watchlist(request: WatchlistRequest, user=Depends(verify_jwt_token)):
    try:
        watchlist_manager.add_ticker(request.user_id, request.ticker, request.tags)
        watchlist = watchlist_manager.get_watchlist(request.user_id)
        return {"watchlist": watchlist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get/{user_id}", response_model=WatchlistResponse)
async def get_watchlist(user_id: str, user=Depends(verify_jwt_token)):
    try:
        watchlist = watchlist_manager.get_watchlist(user_id)
        return {"watchlist": watchlist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/filter/{user_id}/{tag}")
async def filter_watchlist(user_id: str, tag: str, user=Depends(verify_jwt_token)):
    try:
        tickers = watchlist_manager.filter_by_tag(user_id, tag)
        return {"tickers": tickers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
