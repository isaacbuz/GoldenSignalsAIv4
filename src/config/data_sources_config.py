"""
Data Sources Configuration for GoldenSignalsAI
Comprehensive list of recommended data sources with details
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class DataSourceTier(Enum):
    FREE = "free"
    FREEMIUM = "freemium"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class DataCategory(Enum):
    NEWS = "news"
    SOCIAL = "social"
    MARKET = "market"
    ALTERNATIVE = "alternative"
    ECONOMIC = "economic"
    OPTIONS = "options"


@dataclass
class DataSourceConfig:
    name: str
    category: DataCategory
    tier: DataSourceTier
    api_endpoint: str
    features: List[str]
    rate_limit: Optional[str]
    estimated_cost: Optional[str]
    priority: int  # 1-5, where 1 is highest priority
    
    
# Premium Financial News Sources
BLOOMBERG_CONFIG = DataSourceConfig(
    name="Bloomberg Terminal API",
    category=DataCategory.NEWS,
    tier=DataSourceTier.ENTERPRISE,
    api_endpoint="https://api.bloomberg.com/",
    features=[
        "Real-time news sentiment",
        "Earnings transcripts",
        "Analyst ratings",
        "Economic forecasts"
    ],
    rate_limit="Unlimited",
    estimated_cost="$24,000+/year",
    priority=1
)

REFINITIV_CONFIG = DataSourceConfig(
    name="Refinitiv Eikon",
    category=DataCategory.NEWS,
    tier=DataSourceTier.ENTERPRISE,
    api_endpoint="https://api.refinitiv.com/",
    features=[
        "Machine-readable news",
        "Sentiment scores",
        "Entity recognition",
        "ESG data"
    ],
    rate_limit="Based on subscription",
    estimated_cost="$20,000+/year",
    priority=1
)

BENZINGA_CONFIG = DataSourceConfig(
    name="Benzinga Pro API",
    category=DataCategory.NEWS,
    tier=DataSourceTier.PREMIUM,
    api_endpoint="https://api.benzinga.com/api/v2/",
    features=[
        "Real-time newsfeeds",
        "Pre-market movers",
        "Unusual options activity",
        "Analyst ratings"
    ],
    rate_limit="10 requests/second",
    estimated_cost="$2,000/year",
    priority=2
)

# Social Media Sources
STOCKTWITS_CONFIG = DataSourceConfig(
    name="StockTwits API",
    category=DataCategory.SOCIAL,
    tier=DataSourceTier.FREEMIUM,
    api_endpoint="https://api.stocktwits.com/api/2/",
    features=[
        "Sentiment indicators",
        "Trending tickers",
        "Message volume",
        "User watchlists"
    ],
    rate_limit="200 requests/hour",
    estimated_cost="Free - $500/month",
    priority=2
)

DISCORD_CONFIG = DataSourceConfig(
    name="Discord Trading Communities",
    category=DataCategory.SOCIAL,
    tier=DataSourceTier.FREE,
    api_endpoint="Discord.py library",
    features=[
        "Real-time chat monitoring",
        "Sentiment analysis",
        "Trending topics",
        "Community alerts"
    ],
    rate_limit="Based on bot limits",
    estimated_cost="Free",
    priority=3
)

# Market Data Providers
IEX_CLOUD_CONFIG = DataSourceConfig(
    name="IEX Cloud",
    category=DataCategory.MARKET,
    tier=DataSourceTier.FREEMIUM,
    api_endpoint="https://cloud.iexapis.com/stable/",
    features=[
        "Real-time quotes",
        "Historical data",
        "Corporate actions",
        "Options data"
    ],
    rate_limit="Based on plan",
    estimated_cost="$9-$999/month",
    priority=1
)

TRADIER_CONFIG = DataSourceConfig(
    name="Tradier API",
    category=DataCategory.MARKET,
    tier=DataSourceTier.FREEMIUM,
    api_endpoint="https://api.tradier.com/v1/",
    features=[
        "Options chains",
        "Greeks calculation",
        "Streaming quotes",
        "Paper trading"
    ],
    rate_limit="60 requests/minute",
    estimated_cost="Free sandbox, $10+/month live",
    priority=2
)

ALPACA_CONFIG = DataSourceConfig(
    name="Alpaca Markets API",
    category=DataCategory.MARKET,
    tier=DataSourceTier.FREE,
    api_endpoint="https://api.alpaca.markets/",
    features=[
        "Real-time data",
        "Historical bars",
        "Crypto data",
        "Paper trading"
    ],
    rate_limit="200 requests/minute",
    estimated_cost="Free",
    priority=2
)

# Alternative Data Sources
THINKNUM_CONFIG = DataSourceConfig(
    name="Thinknum",
    category=DataCategory.ALTERNATIVE,
    tier=DataSourceTier.ENTERPRISE,
    api_endpoint="https://api.thinknum.com/",
    features=[
        "Web scraping data",
        "Company KPIs",
        "Job postings",
        "Product listings"
    ],
    rate_limit="Based on plan",
    estimated_cost="$3,000+/month",
    priority=3
)

SATELLITE_CONFIG = DataSourceConfig(
    name="RS Metrics",
    category=DataCategory.ALTERNATIVE,
    tier=DataSourceTier.ENTERPRISE,
    api_endpoint="Contact for API",
    features=[
        "Parking lot traffic",
        "Retail foot traffic",
        "Industrial activity",
        "Supply chain monitoring"
    ],
    rate_limit="Custom",
    estimated_cost="$10,000+/month",
    priority=4
)

# Economic Data Sources
FRED_CONFIG = DataSourceConfig(
    name="FRED API",
    category=DataCategory.ECONOMIC,
    tier=DataSourceTier.FREE,
    api_endpoint="https://api.stlouisfed.org/fred/",
    features=[
        "800,000+ time series",
        "GDP data",
        "Interest rates",
        "Employment statistics"
    ],
    rate_limit="120 requests/minute",
    estimated_cost="Free",
    priority=1
)

QUANDL_CONFIG = DataSourceConfig(
    name="Nasdaq Data Link (Quandl)",
    category=DataCategory.ECONOMIC,
    tier=DataSourceTier.FREEMIUM,
    api_endpoint="https://data.nasdaq.com/api/v3/",
    features=[
        "Futures data",
        "Commodities",
        "Economic indicators",
        "Alternative datasets"
    ],
    rate_limit="Based on plan",
    estimated_cost="$0-$2,000/month",
    priority=2
)

# Options Flow Data
FLOWALGO_CONFIG = DataSourceConfig(
    name="FlowAlgo API",
    category=DataCategory.OPTIONS,
    tier=DataSourceTier.PREMIUM,
    api_endpoint="https://api.flowalgo.com/",
    features=[
        "Real-time options flow",
        "Unusual activity alerts",
        "Dark pool prints",
        "Smart money tracking"
    ],
    rate_limit="Based on plan",
    estimated_cost="$200-$500/month",
    priority=2
)

UNUSUAL_WHALES_CONFIG = DataSourceConfig(
    name="Unusual Whales API",
    category=DataCategory.OPTIONS,
    tier=DataSourceTier.PREMIUM,
    api_endpoint="https://api.unusualwhales.com/",
    features=[
        "Congressional trading",
        "Options flow",
        "Whale alerts",
        "ETF flows"
    ],
    rate_limit="Based on plan",
    estimated_cost="$50-$200/month",
    priority=3
)


# Aggregate all data sources
ALL_DATA_SOURCES = [
    # News Sources
    BLOOMBERG_CONFIG,
    REFINITIV_CONFIG,
    BENZINGA_CONFIG,
    
    # Social Sources
    STOCKTWITS_CONFIG,
    DISCORD_CONFIG,
    
    # Market Data
    IEX_CLOUD_CONFIG,
    TRADIER_CONFIG,
    ALPACA_CONFIG,
    
    # Alternative Data
    THINKNUM_CONFIG,
    SATELLITE_CONFIG,
    
    # Economic Data
    FRED_CONFIG,
    QUANDL_CONFIG,
    
    # Options Flow
    FLOWALGO_CONFIG,
    UNUSUAL_WHALES_CONFIG
]


def get_sources_by_category(category: DataCategory) -> List[DataSourceConfig]:
    """Get all data sources for a specific category"""
    return [source for source in ALL_DATA_SOURCES if source.category == category]


def get_sources_by_tier(tier: DataSourceTier) -> List[DataSourceConfig]:
    """Get all data sources for a specific pricing tier"""
    return [source for source in ALL_DATA_SOURCES if source.tier == tier]


def get_priority_sources(max_priority: int = 2) -> List[DataSourceConfig]:
    """Get high priority data sources"""
    return [source for source in ALL_DATA_SOURCES if source.priority <= max_priority]


# Integration priority recommendations
INTEGRATION_PHASES = {
    "Phase 1 - Essential": [
        "IEX Cloud",  # Affordable, high-quality market data
        "FRED API",  # Free economic data
        "StockTwits API",  # Social sentiment
        "Alpaca Markets API"  # Free real-time data
    ],
    "Phase 2 - Enhanced": [
        "Benzinga Pro API",  # Professional news
        "Tradier API",  # Options data
        "Quandl",  # Alternative datasets
        "Unusual Whales API"  # Options flow
    ],
    "Phase 3 - Premium": [
        "Bloomberg Terminal API",  # Enterprise news
        "FlowAlgo API",  # Professional options flow
        "Thinknum",  # Alternative data
    ],
    "Phase 4 - Enterprise": [
        "Refinitiv Eikon",  # Comprehensive analytics
        "RS Metrics",  # Satellite data
    ]
} 