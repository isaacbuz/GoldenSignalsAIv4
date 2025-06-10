import asyncio
from typing import List, Dict

async def fetch_price(venue: str, symbol: str) -> float:
    import random, asyncio
    await asyncio.sleep(0.1)
    return round(100 + random.uniform(-1, 1), 2)

async def scan_symbol(symbol: str, venues: List[str], threshold: float = 0.3) -> Dict:
    tasks = {v: asyncio.create_task(fetch_price(v, symbol)) for v in venues}
    prices = {v: await task for v, task in tasks.items()}

    min_venue = min(prices, key=prices.get)
    max_venue = max(prices, key=prices.get)
    spread = prices[max_venue] - prices[min_venue]

    if spread >= threshold:
        return {
            "symbol": symbol,
            "buy_venue": min_venue,
            "buy_price": prices[min_venue],
            "sell_venue": max_venue,
            "sell_price": prices[max_venue],
            "spread": round(spread, 4)
        }

    return None

async def scan_all(symbols: List[str], venues: List[str]) -> List[Dict]:
    tasks = [scan_symbol(symbol, venues) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]
