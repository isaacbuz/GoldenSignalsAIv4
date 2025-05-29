import datetime
import os
import asyncio
from backend.nlp.sentiment_engine import analyze_sentiment
from typing import List, Dict, Any
import httpx
import logging

# Optional: aioredis for async caching
try:
    import aioredis
    redis_available = True
except ImportError:
    aioredis = None
    redis_available = False

FACEBOOK_ACCESS_TOKEN = os.environ.get("FACEBOOK_ACCESS_TOKEN")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# In-memory fallback cache
in_memory_cache: Dict[str, Any] = {}

logger = logging.getLogger("sentiment_aggregator")

# Placeholder social media fetchers (to be replaced with real API/scraping logic)
async def get_cache(key: str):
    if redis_available:
        try:
            redis = await aioredis.from_url(REDIS_URL)
            val = await redis.get(key)
            if val:
                return val.decode()
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
    return in_memory_cache.get(key)

async def set_cache(key: str, value: Any, ttl: int = 300):
    if redis_available:
        try:
            redis = await aioredis.from_url(REDIS_URL)
            await redis.set(key, value, ex=ttl)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    in_memory_cache[key] = value

async def fetch_posts_from_twitter(symbol: str) -> list[str]:
    key = f"twitter:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    # Placeholder: Replace with real Twitter API logic
    posts = [f"{symbol} is pumping!", f"I'm bullish on {symbol}", f"{symbol} is overvalued"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_linkedin(symbol: str) -> list[str]:
    key = f"linkedin:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    posts = [f"{symbol} earnings report looked good", f"Analyst upgrade on {symbol}"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_reddit(symbol: str) -> list[str]:
    key = f"reddit:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    async with httpx.AsyncClient() as client:
        try:
            url = f"https://www.reddit.com/search.json?q={symbol}&limit=5"
            resp = await client.get(url, headers={"User-Agent": "GoldenSignalsAI/1.0"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = [item['data']['title'] for item in data['data']['children']]
        except Exception as e:
            logger.error(f"[Reddit API] Error: {e}")
            posts = [f"{symbol} is going to the moon", f"Short squeeze incoming on {symbol}"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_stocktwits(symbol: str) -> list[str]:
    key = f"stocktwits:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    async with httpx.AsyncClient() as client:
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = [msg.get("body", "") for msg in data.get("messages", [])][:5]
        except Exception as e:
            logger.error(f"[StockTwits API] Error: {e}")
            posts = [f"{symbol} trending on StockTwits"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_tiktok(symbol: str) -> list[str]:
    key = f"tiktok:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    posts = [f"{symbol} viral on TikTok", f"TikTokers are talking about {symbol}"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_hackernews(symbol: str) -> list[str]:
    key = f"hackernews:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    async with httpx.AsyncClient() as client:
        try:
            url = f"https://hn.algolia.com/api/v1/search?query={symbol}&tags=story"
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = [hit.get("title", "") for hit in data.get("hits", [])][:5]
        except Exception as e:
            logger.error(f"[HackerNews API] Error: {e}")
            posts = [f"{symbol} mentioned on Hacker News"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_facebook(symbol: str) -> list[str]:
    """
    Async: Fetch recent Facebook posts mentioning the symbol using Graph API.
    Requires FACEBOOK_ACCESS_TOKEN in environment. Returns up to 5 post messages.
    """
    key = f"facebook:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    if not FACEBOOK_ACCESS_TOKEN:
        posts = [f"{symbol} is trending on Facebook", f"Saw lots of posts about {symbol}"]
        await set_cache(key, posts)
        return posts
    async with httpx.AsyncClient() as client:
        try:
            url = f"https://graph.facebook.com/v19.0/search"
            params = {
                "q": symbol,
                "type": "post",
                "fields": "message",
                "access_token": FACEBOOK_ACCESS_TOKEN,
                "limit": 5
            }
            resp = await client.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = [item.get("message", "") for item in data.get("data", []) if item.get("message")]
        except Exception as e:
            logger.error(f"[Facebook API] Error: {e}")
            posts = [f"{symbol} is trending on Facebook", f"Saw lots of posts about {symbol}"]
    await set_cache(key, posts)
    return posts

async def fetch_posts_from_youtube(symbol: str) -> list[str]:
    """
    Async: Fetch recent YouTube video titles and descriptions mentioning the symbol.
    Requires YOUTUBE_API_KEY in environment. Returns up to 5 video snippets.
    """
    key = f"youtube:{symbol}"
    cached = await get_cache(key)
    if cached:
        return cached
    if not YOUTUBE_API_KEY:
        posts = [f"{symbol} stock analysis video", f"YouTube influencer bullish on {symbol}"]
        await set_cache(key, posts)
        return posts
    async with httpx.AsyncClient() as client:
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": symbol,
                "type": "video",
                "maxResults": 5,
                "key": YOUTUBE_API_KEY
            }
            resp = await client.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                title = snippet.get("title", "")
                desc = snippet.get("description", "")
                if title:
                    results.append(title)
                if desc:
                    results.append(desc)
            posts = results[:5] if results else [f"{symbol} stock analysis video"]
        except Exception as e:
            logger.error(f"[YouTube API] Error: {e}")
            posts = [f"{symbol} stock analysis video", f"YouTube influencer bullish on {symbol}"]
    await set_cache(key, posts)
    return posts

async def fetch_all_sentiment(symbol: str) -> Dict:
    """
    Run all platform fetchers in parallel, aggregate sentiment, and (optionally) persist to DB.
    Returns sample posts per platform as well.
    """
    fetchers = [
        fetch_posts_from_twitter,
        fetch_posts_from_linkedin,
        fetch_posts_from_reddit,
        fetch_posts_from_stocktwits,
        fetch_posts_from_tiktok,
        fetch_posts_from_hackernews,
        fetch_posts_from_facebook,
        fetch_posts_from_youtube,
    ]
    platform_names = [
        "X",
        "LinkedIn",
        "Reddit",
        "StockTwits",
        "TikTok",
        "HackerNews",
        "Facebook",
        "YouTube"
    ]
    # Run all fetchers concurrently
    post_lists = await asyncio.gather(*[fetcher(symbol) for fetcher in fetchers])
    posts = dict(zip(platform_names, post_lists))

    scores = {}
    total = 0
    count = 0
    sample_post = {}
    for platform, texts in posts.items():
        polarity_scores = [analyze_sentiment(t)["score"] for t in texts]
        avg_score = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0
        scores[platform] = avg_score
        total += sum(polarity_scores)
        count += len(polarity_scores)
        if texts:
            sample_post[platform] = texts[0]

    final_score = total / count if count else 0
    trend = "bullish" if final_score >= 0.2 else "bearish" if final_score <= -0.2 else "neutral"

    # Optionally persist to DB (pseudo, replace with real DB logic if available)
    try:
        from backend.db.models import SentimentScore
        from backend.db.session import get_db
        db = next(get_db())
        db.add(SentimentScore(
            symbol=symbol,
            score=final_score,
            platform_breakdown=scores,
            trend=trend,
            sample_post=sample_post,
            updated_at=datetime.datetime.utcnow()
        ))
        db.commit()
    except Exception as e:
        logger.info(f"[DB] Sentiment persistence skipped or failed: {e}")

    return {
        "symbol": symbol,
        "score": round(final_score, 3),
        "platform_breakdown": {k: round(v, 3) for k, v in scores.items()},
        "trend": trend,
        "sample_post": sample_post,
        "updated_at": datetime.datetime.utcnow().isoformat()
    }

def recommend_top_sentiment_stocks(symbols: List[str], limit: int = 5, direction: str = "bullish"):
    results = [fetch_all_sentiment(sym) for sym in symbols]
    filtered = [r for r in results if r["trend"] == direction]
    return sorted(filtered, key=lambda x: abs(x["score"]), reverse=True)[:limit]
