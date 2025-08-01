#!/usr/bin/env python3
"""
Live Transformer Model Evaluation
Connects to live market data and evaluates transformer model performance in real-time
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from agents.ml.transformer_agent import TransformerAgent
from src.ml.models.market_data import MarketData
from src.data.fetchers.live_data_fetcher import (
    UnifiedDataFeed,
    YahooFinanceSource,
    PolygonIOSource,
    MarketData as LiveMarketData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTransformerEvaluator:
    """Real-time evaluation of transformer model performance."""

    def __init__(
        self,
        model_path: str = "models/transformer",
        results_dir: str = "ml_training/metrics",
        symbols: List[str] = ['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ'],
        evaluation_window: int = 100  # Number of predictions to keep for evaluation
    ):
        self.model_path = Path(model_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols
        self.evaluation_window = evaluation_window

        # Initialize transformer agent
        self.agent = TransformerAgent()

        # Load model if exists
        if self.model_path.exists():
            self.agent.model.load_state_dict(
                torch.load(self.model_path / "transformer_model.pth")
            )
            logger.info("✅ Loaded existing transformer model")

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.model.to(self.device)

        # Initialize data feed
        self.data_feed = UnifiedDataFeed(primary_source="yahoo")

        # Initialize metrics storage
        self.metrics = {
            symbol: {
                'predictions': [],
                'actuals': [],
                'confidences': [],
                'timestamps': []
            } for symbol in symbols
        }

        # Performance tracking
        self.performance_history = {
            symbol: {
                'mse': [],
                'rmse': [],
                'r2': [],
                'direction_accuracy': [],
                'average_confidence': []
            } for symbol in symbols
        }

    async def setup_data_sources(self):
        """Set up data sources based on available API keys."""
        # Add Yahoo Finance (free)
        yahoo_source = YahooFinanceSource()
        self.data_feed.add_source("yahoo", yahoo_source)
        logger.info("✅ Added Yahoo Finance data source")

        # Add Polygon if API key is available
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key:
            polygon_source = PolygonIOSource(polygon_key)
            self.data_feed.add_source("polygon", polygon_source)
            logger.info("✅ Added Polygon.io data source")

        # Register callbacks
        self.data_feed.register_callback('quote', self.on_quote_update)

    async def on_quote_update(self, data: LiveMarketData):
        """Handle real-time quote updates."""
        symbol = data.symbol

        # Update metrics storage
        self.metrics[symbol]['actuals'].append(data.price)
        self.metrics[symbol]['timestamps'].append(data.timestamp)

        # Keep only last N predictions
        for key in ['actuals', 'predictions', 'confidences', 'timestamps']:
            self.metrics[symbol][key] = self.metrics[symbol][key][-self.evaluation_window:]

        # Generate prediction
        market_data = MarketData(
            timestamp=[data.timestamp],
            open=[data.open],
            high=[data.high],
            low=[data.low],
            close=[data.close],
            volume=[data.volume]
        )

        result = self.agent.process_signal(market_data)

        # Store prediction
        self.metrics[symbol]['predictions'].append(result['prediction'])
        self.metrics[symbol]['confidences'].append(result['confidence'])

        # Calculate and store metrics
        metrics = self._calculate_metrics(symbol)
        for metric, value in metrics.items():
            self.performance_history[symbol][metric].append(value)

        # Log performance
        logger.info(
            f"📊 {symbol} - "
            f"Price: ${data.price:.2f}, "
            f"Prediction: {result['prediction']:.4f}, "
            f"Confidence: {result['confidence']:.2f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"Direction Accuracy: {metrics['direction_accuracy']:.2f}"
        )

        # Generate plots periodically
        if len(self.metrics[symbol]['predictions']) % 10 == 0:  # Every 10 predictions
            self.plot_results(symbol)

    def _calculate_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate performance metrics for a symbol."""
        predictions = np.array(self.metrics[symbol]['predictions'])
        actuals = np.array(self.metrics[symbol]['actuals'])
        confidences = np.array(self.metrics[symbol]['confidences'])

        if len(predictions) < 2:
            return {
                'mse': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'direction_accuracy': 0.0,
                'average_confidence': np.mean(confidences)
            }

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)

        # Direction accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(np.diff(actuals))
        direction_accuracy = accuracy_score(actual_direction, pred_direction[:-1])

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'average_confidence': np.mean(confidences)
        }

    def plot_results(self, symbol: str):
        """Generate performance visualizations for a symbol."""
        logger.info(f"📈 Generating plots for {symbol}...")

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)

        # 1. Predictions vs Actual
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.metrics[symbol]['timestamps'],
                self.metrics[symbol]['actuals'],
                label='Actual', color='blue', alpha=0.7)
        ax1.plot(self.metrics[symbol]['timestamps'],
                self.metrics[symbol]['predictions'],
                label='Predicted', color='red', alpha=0.7)
        ax1.set_title(f'{symbol} - Predictions vs Actual')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)

        # 2. Confidence Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(self.metrics[symbol]['confidences'], bins=20, ax=ax2)
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')

        # 3. Prediction Error Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        errors = np.array(self.metrics[symbol]['predictions']) - np.array(self.metrics[symbol]['actuals'])
        sns.histplot(errors, bins=20, ax=ax3)
        ax3.set_title('Prediction Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Count')

        # 4. Performance Metrics Over Time
        ax4 = fig.add_subplot(gs[2, :])
        metrics_df = pd.DataFrame(self.performance_history[symbol])
        metrics_df.plot(ax=ax4)
        ax4.set_title('Performance Metrics Over Time')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True)

        # Save plot
        plt.tight_layout()
        plt.savefig(self.results_dir / f'transformer_live_{symbol}.png')
        plt.close()

        # Save metrics
        self._save_metrics(symbol)

    def _save_metrics(self, symbol: str):
        """Save current metrics to file."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'current_metrics': self._calculate_metrics(symbol),
            'performance_history': self.performance_history[symbol]
        }

        with open(self.results_dir / f'transformer_live_{symbol}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    async def start(self):
        """Start live evaluation."""
        try:
            logger.info("🚀 Starting Live Transformer Model Evaluation...")

            # Set up data sources
            await self.setup_data_sources()

            # Connect data feed
            await self.data_feed.connect_all()

            # Start streaming
            await self.data_feed.start_streaming(self.symbols, interval=5)  # 5 second updates

        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.data_feed.stop_streaming()

async def main():
    """Main evaluation script."""
    # Initialize evaluator
    evaluator = LiveTransformerEvaluator()

    # Start evaluation
    await evaluator.start()

if __name__ == "__main__":
    asyncio.run(main())
