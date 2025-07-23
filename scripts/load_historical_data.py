#!/usr/bin/env python3
"""
Script to load 30 years of historical data
Run this to populate the database with historical market data
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.historical_data_service import historical_data_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_initial_data():
    """Load initial historical data for key symbols"""

    # Key symbols to load
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'SPY', 'QQQ', 'DIA', 'IWM',  # ETFs
        'GLD', 'TLT', 'VXX'  # Others
    ]

    logger.info("Starting historical data load...")

    # Initialize database schema
    await historical_data_service.initialize_database()

    # Load stock data
    logger.info(f"Loading 30 years of data for {len(symbols)} symbols...")
    for symbol in symbols:
        try:
            logger.info(f"Loading {symbol}...")
            await historical_data_service.load_stock_history(symbol)
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    # Load economic indicators
    logger.info("Loading economic indicators from FRED...")
    try:
        await historical_data_service.load_economic_indicators()
    except Exception as e:
        logger.error(f"Failed to load economic indicators: {e}")

    logger.info("Historical data load complete!")

    # Show some statistics
    stats = await historical_data_service.get_performance_stats('AAPL', 30)
    logger.info(f"Sample stats for AAPL: {stats}")


async def quick_demo():
    """Quick demonstration of historical data capabilities"""

    # Example: Find similar market conditions
    logger.info("\n=== Finding Similar Historical Conditions ===")

    current_conditions = {
        'rsi': 65,
        'volume_ratio': 1.5,
        'volatility': 0.025,
        'trend': 0.03
    }

    similar = await historical_data_service.query_similar_market_conditions(
        'AAPL',
        current_conditions,
        lookback_days=365
    )

    if similar:
        logger.info(f"Found {len(similar)} similar historical setups:")
        for setup in similar[:3]:
            logger.info(f"  Date: {setup['date']}, Next week return: {setup.get('next_week_return', 'N/A'):.2%}")

    # Example: Get regime analysis
    logger.info("\n=== Current Market Regime ===")
    regime = await historical_data_service.get_regime_analysis('AAPL')
    logger.info(f"Current regime: {regime}")

    # Example: Get long-term performance
    logger.info("\n=== Long-term Performance Stats ===")
    stats = await historical_data_service.get_performance_stats('AAPL', 10)
    logger.info(f"10-year stats: {stats}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load historical market data')
    parser.add_argument('--full', action='store_true', help='Load full 30 years of data')
    parser.add_argument('--demo', action='store_true', help='Run quick demo')
    parser.add_argument('--symbol', type=str, help='Load data for specific symbol')

    args = parser.parse_args()

    if args.symbol:
        # Load single symbol
        async def load_single():
            await historical_data_service.initialize_database()
            await historical_data_service.load_stock_history(args.symbol)
            stats = await historical_data_service.get_performance_stats(args.symbol, 30)
            logger.info(f"Stats for {args.symbol}: {stats}")

        asyncio.run(load_single())
    elif args.demo:
        asyncio.run(quick_demo())
    else:
        asyncio.run(load_initial_data())
