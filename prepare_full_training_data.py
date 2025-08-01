#!/usr/bin/env python3
"""
Prepare Full Training Dataset - 20 Years of Historical Data
"""

import asyncio
import os
import sys
from datetime import datetime
import logging
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.live_data_connector import live_data_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Major stocks and ETFs for comprehensive training
TRAINING_SYMBOLS = [
    # Major Indices & ETFs
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI',

    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',

    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS',

    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',

    # Consumer
    'WMT', 'HD', 'DIS', 'NKE', 'MCD',

    # Energy
    'XOM', 'CVX', 'COP',

    # Industrial
    'BA', 'CAT', 'GE', 'MMM',

    # Semiconductors
    'AMD', 'INTC', 'MU', 'QCOM',

    # Other Notable
    'V', 'MA', 'PYPL', 'SQ', 'NFLX', 'CRM', 'ADBE'
]

# Features to extract
TECHNICAL_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
    'BB_upper', 'BB_lower', 'BB_width',
    'ATR', 'Volume_ratio'
]


async def prepare_full_dataset():
    """Prepare 20 years of training data"""
    print("\n🚀 Starting Full Training Data Preparation")
    print(f"📊 Symbols to process: {len(TRAINING_SYMBOLS)}")
    print(f"📅 Time period: 20 years")
    print("=" * 60)

    try:
        # Initialize connector
        print("\n🔌 Initializing database connections...")
        await live_data_connector.initialize()
        print("✅ Database connections established!")

        # Create progress tracking
        total_symbols = len(TRAINING_SYMBOLS)
        processed = 0
        failed = []

        # Process in batches to avoid overwhelming APIs
        batch_size = 5
        for i in range(0, len(TRAINING_SYMBOLS), batch_size):
            batch = TRAINING_SYMBOLS[i:i + batch_size]

            print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(TRAINING_SYMBOLS) + batch_size - 1)//batch_size}")
            print(f"   Symbols: {', '.join(batch)}")

            # Prepare dataset for batch
            dataset = await live_data_connector.prepare_training_dataset(
                symbols=batch,
                years=20,
                features=TECHNICAL_FEATURES
            )

            if not dataset.empty:
                processed += len(batch)
                print(f"✅ Batch completed: {len(dataset)} samples collected")
            else:
                failed.extend(batch)
                print(f"⚠️  Batch failed or returned no data")

            # Small delay between batches to respect rate limits
            if i + batch_size < len(TRAINING_SYMBOLS):
                print("⏳ Waiting 2 seconds before next batch...")
                await asyncio.sleep(2)

        # Summary
        print("\n" + "=" * 60)
        print("📊 TRAINING DATA PREPARATION SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed: {processed}/{total_symbols} symbols")

        if failed:
            print(f"❌ Failed symbols: {', '.join(failed)}")

        # Check saved dataset
        if os.path.exists('data/training_dataset.parquet'):
            import pandas as pd
            final_dataset = pd.read_parquet('data/training_dataset.parquet')

            print(f"\n📈 Final Dataset Statistics:")
            print(f"   Total samples: {len(final_dataset):,}")
            print(f"   Date range: {final_dataset.index.min()} to {final_dataset.index.max()}")
            print(f"   Features: {len(final_dataset.columns)}")
            print(f"   Symbols: {final_dataset['symbol'].nunique()}")

            # Save summary
            summary = {
                'preparation_date': datetime.now().isoformat(),
                'total_samples': len(final_dataset),
                'symbols_processed': processed,
                'symbols_failed': failed,
                'features': final_dataset.columns.tolist(),
                'date_range': {
                    'start': str(final_dataset.index.min()),
                    'end': str(final_dataset.index.max())
                }
            }

            with open('data/training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            print("\n✅ Training data preparation completed successfully!")
            print("📄 Summary saved to: data/training_summary.json")
            print("📊 Dataset saved to: data/training_dataset.parquet")

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        print(f"\n❌ Data preparation failed: {e}")

    finally:
        # Close connections
        await live_data_connector.close()


async def verify_data_quality():
    """Verify the quality of prepared data"""
    print("\n🔍 Verifying Data Quality...")

    if os.path.exists('data/training_dataset.parquet'):
        import pandas as pd
        df = pd.read_parquet('data/training_dataset.parquet')

        print(f"\n📊 Data Quality Report:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate rows: {df.duplicated().sum()}")

        # Check each symbol
        print("\n📈 Per-Symbol Statistics:")
        for symbol in df['symbol'].unique()[:10]:  # Show first 10
            symbol_data = df[df['symbol'] == symbol]
            print(f"   {symbol}: {len(symbol_data)} samples, "
                  f"from {symbol_data.index.min()} to {symbol_data.index.max()}")
    else:
        print("❌ No dataset found to verify")


async def main():
    """Main execution"""
    print("🏦 GoldenSignalsAI - Full Training Data Preparation")
    print("📅 Preparing 20 years of historical market data")
    print("=" * 60)

    # Prepare the dataset
    await prepare_full_dataset()

    # Verify data quality
    await verify_data_quality()

    print("\n✅ Process completed!")
    print("\n📝 Next steps:")
    print("1. Review the data quality report above")
    print("2. Update simple_backend.py to use live data connector")
    print("3. Train ML models using the prepared dataset")
    print("4. Run backtesting with historical data")


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Run the preparation
    asyncio.run(main())
