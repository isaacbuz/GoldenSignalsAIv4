from fastapi import APIRouter
import random

router = APIRouter()

@router.get("/api/etf/heatmap")
def etf_heatmap():
    etfs = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV"]
    data = []
    for symbol in etfs:
        nav = round(random.uniform(390, 430), 2)
        price = nav + random.uniform(-2.5, 2.5)
        delta = round(price - nav, 2)
        data.append({
            "symbol": symbol,
            "nav": nav,
            "price": price,
            "delta": delta
        })
    return data
