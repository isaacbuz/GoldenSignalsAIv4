# create_part2b.py
# Purpose: Creates files in the application/services/ directory for the GoldenSignalsAI project,
# including core services like signal generation, backtesting, and risk management.
# Incorporates improvements for options trading with volatility-based algorithms and risk management.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part2b():
    """Create files in application/services/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating application/services files in {base_dir}"})

    # Define services directory files
    services_files = {
        "application/services/__init__.py": """# application/services/__init__.py
# Purpose: Marks the services directory as a Python subpackage, enabling imports
# for core services like signal generation and risk management.
""",
        "application/services/custom_algorithm.py": """# application/services/custom_algorithm.py
# Purpose: Implements a custom trading algorithm combining volatility breakouts,
# momentum, and options flow signals for options trading strategies.

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class CustomAlgorithm:
    \"\"\"Custom trading algorithm for generating options trading signals.\"\"\"
    def __init__(self, volatility_window: int = 20, momentum_window: int = 10, iv_threshold: float = 0.1):
        \"\"\"Initialize the CustomAlgorithm.
        
        Args:
            volatility_window (int): Window for volatility calculation.
            momentum_window (int): Window for momentum calculation.
            iv_threshold (float): Threshold for implied volatility signal.
        \"\"\"
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.iv_threshold = iv_threshold
        logger.info({
            "message": "CustomAlgorithm initialized",
            "volatility_window": volatility_window,
            "momentum_window": momentum_window,
            "iv_threshold": iv_threshold
        })

    def calculate_volatility_breakout(self, stock_data: pd.DataFrame) -> float:
        \"\"\"Calculate volatility breakout signal.
        
        Args:
            stock_data (pd.DataFrame): Stock data with 'close' column.
        
        Returns:
            float: Volatility breakout signal (1 for buy, -1 for sell, 0 for hold).
        \"\"\"
        logger.debug({"message": "Calculating volatility breakout"})
        try:
            prices = stock_data['close']
            volatility = prices.pct_change().rolling(self.volatility_window).std() * np.sqrt(252)
            latest_volatility = volatility.iloc[-1]
            avg_volatility = volatility[-self.volatility_window:].mean()

            if latest_volatility > avg_volatility * 1.5:
                return 1  # Buy signal (volatility spike)
            elif latest_volatility < avg_volatility * 0.5:
                return -1  # Sell signal (volatility contraction)
            return 0
        except Exception as e:
            logger.error({"message": f"Failed to calculate volatility breakout: {str(e)}"})
            return 0

    def calculate_momentum(self, stock_data: pd.DataFrame) -> float:
        \"\"\"Calculate momentum signal.
        
        Args:
            stock_data (pd.DataFrame): Stock data with 'close' column.
        
        Returns:
            float: Momentum signal (1 for buy, -1 for sell, 0 for hold).
        \"\"\"
        logger.debug({"message": "Calculating momentum"})
        try:
            prices = stock_data['close']
            momentum = (prices.iloc[-1] - prices.iloc[-self.momentum_window]) / prices.iloc[-self.momentum_window]
            
            if momentum > 0.05:
                return 1  # Buy signal (positive momentum)
            elif momentum < -0.05:
                return -1  # Sell signal (negative momentum)
            return 0
        except Exception as e:
            logger.error({"message": f"Failed to calculate momentum: {str(e)}"})
            return 0

    def calculate_options_flow_signal(self, options_data: pd.DataFrame) -> float:
        \"\"\"Calculate options flow signal based on implied volatility and volume.
        
        Args:
            options_data (pd.DataFrame): Options data with 'call_volume', 'put_volume', 'iv'.
        
        Returns:
            float: Options flow signal (1 for buy, -1 for sell, 0 for hold).
        \"\"\"
        logger.debug({"message": "Calculating options flow signal"})
        try:
            call_volume = options_data[options_data['call_put'] == 'call']['call_volume'].sum()
            put_volume = options_data[options_data['call_put'] == 'put']['put_volume'].sum()
            call_iv = options_data[options_data['call_put'] == 'call']['iv'].mean()
            put_iv = options_data[options_data['call_put'] == 'put']['iv'].mean()
            iv_skew = call_iv - put_iv

            if iv_skew > self.iv_threshold and call_volume > put_volume:
                return 1  # Buy signal (bullish options flow)
            elif iv_skew < -self.iv_threshold and put_volume > call_volume:
                return -1  # Sell signal (bearish options flow)
            return 0
        except Exception as e:
            logger.error({"message": f"Failed to calculate options flow signal: {str(e)}"})
            return 0

    def generate_signal(self, stock_data: pd.DataFrame, options_data: pd.DataFrame) -> Dict:
        \"\"\"Generate a trading signal by combining volatility, momentum, and options flow.
        
        Args:
            stock_data (pd.DataFrame): Historical stock data.
            options_data (pd.DataFrame): Options chain data.
        
        Returns:
            Dict: Trading signal with 'action' and 'confidence'.
        \"\"\"
        logger.info({"message": "Generating trading signal"})
        try:
            volatility_signal = self.calculate_volatility_breakout(stock_data)
            momentum_signal = self.calculate_momentum(stock_data)
            options_signal = self.calculate_options_flow_signal(options_data)

            # Combine signals with weights
            weights = {'volatility': 0.4, 'momentum': 0.3, 'options': 0.3}
            combined_signal = (
                weights['volatility'] * volatility_signal +
                weights['momentum'] * momentum_signal +
                weights['options'] * options_signal
            )
            confidence = abs(combined_signal)

            action = "buy" if combined_signal > 0.3 else "sell" if combined_signal < -0.3 else "hold"
            signal = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "volatility_signal": volatility_signal,
                    "momentum_signal": momentum_signal,
                    "options_signal": options_signal
                }
            }
            logger.info({"message": f"Generated signal: {signal}"})
            return signal
        except Exception as e:
            logger.error({"message": f"Failed to generate signal: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}
""",
        "application/services/signal_engine.py": """# application/services/signal_engine.py
# Purpose: Generates trading signals by combining AI predictions with technical indicators,
# tailored for options trading with regime-adjusted signals.

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from domain.trading.indicators import Indicators
from application.ai_service.orchestrator import Orchestrator

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class SignalEngine:
    \"\"\"Generates trading signals combining AI predictions and technical indicators.\"\"\"
    def __init__(self, data_fetcher, user_id: str, risk_profile: str = "balanced"):
        \"\"\"Initialize the SignalEngine.
        
        Args:
            data_fetcher: Data fetcher instance for market data.
            user_id (str): User identifier for personalization.
            risk_profile (str): User risk profile ('conservative', 'balanced', 'aggressive').
        \"\"\"
        self.data_fetcher = data_fetcher
        self.user_id = user_id
        self.risk_profile = risk_profile
        self.orchestrator = Orchestrator(data_fetcher)
        logger.info({
            "message": "SignalEngine initialized",
            "user_id": user_id,
            "risk_profile": risk_profile
        })

    async def generate_signal(self, symbol: str) -> Dict:
        \"\"\"Generate a trading signal for a given symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL').
        
        Returns:
            Dict: Trading signal with action, confidence, and metadata.
        \"\"\"
        logger.info({"message": f"Generating signal for {symbol}"})
        try:
            # Fetch market data
            market_data = await self.orchestrator.consume_stream(symbol)
            if not market_data:
                logger.error({"message": f"No market data available for {symbol}"})
                return {"symbol": symbol, "action": "hold", "confidence": 0.0, "metadata": {}}

            stock_data = market_data['stock_data']
            options_data = market_data['options_data']
            news_articles = market_data['news_articles']

            # Prepare data for AI models
            X_new_lstm = np.array(stock_data['Close'].tail(60)).reshape(1, 60, 1)  # Simplified input
            X_new_tree = stock_data.tail(10)

            # Generate AI prediction
            ai_score = await self.orchestrator.ensemble_predict(symbol, X_new_lstm, X_new_tree)
            if ai_score is None:
                logger.error({"message": f"AI prediction failed for {symbol}"})
                ai_score = 0.0

            # Generate technical indicator signals
            indicators = Indicators(stock_data)
            indicator_signals = indicators.compute_regime_adjusted_signal(['RSI', 'MACD'])
            indicator_score = sum(indicator_signals.values()) / len(indicator_signals) if indicator_signals else 0.0

            # Analyze sentiment from news
            sentiment_score = self.orchestrator.analyze_sentiment(news_articles)

            # Combine scores
            weights = {'ai': 0.5, 'indicators': 0.3, 'sentiment': 0.2}
            final_score = (
                weights['ai'] * ai_score +
                weights['indicators'] * indicator_score +
                weights['sentiment'] * sentiment_score
            )
            confidence = abs(final_score)

            # Adjust signal based on risk profile for options trading
            if self.risk_profile == "conservative":
                confidence *= 0.7
            elif self.risk_profile == "aggressive":
                confidence *= 1.3

            action = "buy" if final_score > 0.3 else "sell" if final_score < -0.3 else "hold"
            regime = indicators.detect_market_regime()

            signal = {
                "symbol": symbol,
                "action": action,
                "confidence": min(confidence, 1.0),
                "ai_score": ai_score,
                "indicator_score": indicator_score,
                "final_score": final_score,
                "timestamp": pd.Timestamp.now().isoformat(),
                "risk_profile": self.risk_profile,
                "indicators": list(indicator_signals.keys()),
                "metadata": {
                    "regime": regime,
                    "sentiment_score": sentiment_score
                }
            }
            logger.info({"message": f"Generated signal for {symbol}: {signal}"})
            return signal
        except Exception as e:
            logger.error({"message": f"Failed to generate signal for {symbol}: {str(e)}"})
            return {"symbol": symbol, "action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}
""",
        "application/services/backtest.py": """# application/services/backtest.py
# Purpose: Implements backtesting logic for trading strategies, used by agents
# for research and validation of options trading strategies.

import pandas as pd
import numpy as np
import logging

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class Backtester:
    \"\"\"Performs backtesting of trading strategies.\"\"\"
    def __init__(self, historical_data: pd.DataFrame, signals: pd.DataFrame):
        \"\"\"Initialize the Backtester.
        
        Args:
            historical_data (pd.DataFrame): Historical stock data with 'close', 'volume'.
            signals (pd.DataFrame): Trading signals with 'symbol', 'timestamp', 'action', 'confidence'.
        \"\"\"
        self.historical_data = historical_data
        self.signals = signals
        logger.info({
            "message": f"Backtester initialized with {len(historical_data)} data points and {len(signals)} signals"
        })

    def run(self, initial_capital: float = 10000) -> dict:
        \"\"\"Run a backtest using the provided signals and historical data.
        
        Args:
            initial_capital (float): Starting capital (default: 10000).
        
        Returns:
            dict: Backtest results including equity curve and performance metrics.
        \"\"\"
        logger.info({"message": f"Running backtest with initial capital={initial_capital:.2f}"})
        try:
            # Align signals with historical data by timestamp
            signals = self.signals.set_index('timestamp')
            data = self.historical_data.copy()
            data['signal'] = signals['action'].reindex(data.index, fill_value='hold')

            # Initialize positions (1 for buy, -1 for sell, 0 for hold)
            positions = pd.Series(0, index=data.index)
            positions[data['signal'] == 'buy'] = 1
            positions[data['signal'] == 'sell'] = -1

            # Calculate returns
            returns = data['close'].pct_change().shift(-1) * positions
            returns = returns.fillna(0)

            # Compute equity curve
            equity = initial_capital * (1 + returns).cumprod()
            total_return = (equity.iloc[-1] - initial_capital) / initial_capital

            # Calculate Sharpe ratio (annualized)
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0

            # Calculate max drawdown
            rolling_max = equity.cummax()
            drawdowns = (equity - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Calculate win/loss ratio
            win_trades = (returns > 0).sum()
            loss_trades = (returns < 0).sum()
            win_loss_ratio = win_trades / loss_trades if loss_trades > 0 else float('inf')

            results = {
                "equity": equity.tolist(),
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_loss_ratio": win_loss_ratio
            }
            logger.info({
                "message": f"Backtest completed: Total Return={total_return*100:.2f}%, Sharpe Ratio={sharpe_ratio:.2f}, Max Drawdown={max_drawdown*100:.2f}%"
            })
            return results
        except Exception as e:
            logger.error({"message": f"Failed to run backtest: {str(e)}"})
            return {"error": str(e)}
""",
        "application/services/risk_manager.py": """# application/services/risk_manager.py
# Purpose: Implements risk management logic for options trading, including position sizing
# using the Kelly Criterion and volatility-based risk adjustments.

import pandas as pd
import numpy as np
import logging
from typing import Dict

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class RiskManager:
    \"\"\"Manages risk for trading strategies with focus on options trading.\"\"\"
    def __init__(self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.1):
        \"\"\"Initialize the RiskManager.
        
        Args:
            max_risk_per_trade (float): Maximum risk per trade as a fraction of capital.
            max_portfolio_risk (float): Maximum portfolio risk as a fraction of capital.
        \"\"\"
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        logger.info({
            "message": "RiskManager initialized",
            "max_risk_per_trade": max_risk_per_trade,
            "max_portfolio_risk": max_portfolio_risk
        })

    def calculate_kelly_position(self, confidence: float, current_price: float) -> float:
        \"\"\"Calculate position size using the Kelly Criterion.
        
        Args:
            confidence (float): Confidence score of the trade signal (0 to 1).
            current_price (float): Current price of the asset.
        
        Returns:
            float: Position size (number of shares/contracts).
        \"\"\"
        logger.debug({"message": f"Calculating Kelly position with confidence={confidence}"})
        try:
            # Simplified Kelly Criterion: f = p - (1-p)/R, where p is win probability, R is win/loss ratio
            win_probability = confidence
            win_loss_ratio = 2.0  # Assumed win/loss ratio
            kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
            kelly_fraction = max(min(kelly_fraction, self.max_risk_per_trade), 0)

            # Position size based on current price
            position_size = kelly_fraction * 10000 / current_price  # Assuming 10,000 capital
            logger.debug({"message": f"Kelly position size: {position_size:.2f}"})
            return position_size
        except Exception as e:
            logger.error({"message": f"Failed to calculate Kelly position: {str(e)}"})
            return 0.0

    def evaluate(self, trade: Dict, stock_data: pd.DataFrame, options_data: pd.DataFrame) -> bool:
        \"\"\"Evaluate if a trade is within risk limits.
        
        Args:
            trade (Dict): Proposed trade with 'symbol', 'action', 'size'.
            stock_data (pd.DataFrame): Stock data.
            options_data (pd.DataFrame): Options data.
        
        Returns:
            bool: True if trade is safe, False otherwise.
        \"\"\"
        logger.info({"message": f"Evaluating trade risk for {trade['symbol']}"})
        try:
            # Calculate portfolio volatility
            returns = stock_data['close'].pct_change().dropna()
            portfolio_volatility = returns.std() * np.sqrt(252)

            # Check if portfolio risk exceeds limit
            if portfolio_volatility > self.max_portfolio_risk:
                logger.warning({"message": f"Portfolio volatility {portfolio_volatility:.2f} exceeds limit {self.max_portfolio_risk}"})
                return False

            # Check position size
            if trade['size'] * stock_data['close'].iloc[-1] > self.max_risk_per_trade * 10000:  # Assuming 10,000 capital
                logger.warning({"message": f"Position size exceeds risk limit for {trade['symbol']}"})
                return False

            logger.info({"message": f"Trade is within risk limits for {trade['symbol']}"})
            return True
        except Exception as e:
            logger.error({"message": f"Failed to evaluate trade risk: {str(e)}"})
            return False
""",
    }

    # Write services directory files
    for file_path, content in services_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 2b: application/services/ created successfully")


if __name__ == "__main__":
    create_part2b()
