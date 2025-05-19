<<<<<<< HEAD
"""
reversion.py
Purpose: Implements a ReversionAgent that identifies mean-reversion opportunities, suitable for options trading strategies like straddles in mean-reverting markets. Integrates with the GoldenSignalsAI agent framework.
"""
=======
# agents/predictive/reversion.py
# Purpose: Implements a ReversionAgent that identifies mean-reversion opportunities,
# suitable for options trading strategies like straddles in mean-reverting markets.
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b

import logging

import pandas as pd
from typing import Dict, Any

from ..base_agent import BaseAgent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class ReversionAgent(BaseAgent):
<<<<<<< HEAD
    """
    Agent that identifies mean-reversion opportunities for swing or day trading.

    Available built-in factors for custom_factors:
        - "rsi": ReversionAgent.rsi_factor
        - "momentum": ReversionAgent.momentum_factor
        - "volume_spike": ReversionAgent.volume_spike_factor
        - "macd": ReversionAgent.macd_factor
        - "bollinger": ReversionAgent.bollinger_bands_factor
    """

    def __init__(self, trade_horizon: str = "swing", mean_reversion_window: int = None, z_score_threshold: float = 2.0, use_volatility_adjustment: bool = True, custom_factors: Dict[str, Any] = None):
        """Initialize the ReversionAgent.

        Args:
            trade_horizon (str): 'day' for intraday/day trading, 'swing' for swing trading (default).
            mean_reversion_window (int): Lookback period for mean-reversion calculation; defaults depend on horizon.
            z_score_threshold (float): Threshold for z-score calculation.
            use_volatility_adjustment (bool): Whether to adapt window/threshold based on volatility.
            custom_factors (dict): Optional dict of factor functions (e.g. {'rsi': rsi_func, 'momentum': mom_func, 'volume_spike': volume_spike_func}).
                See: ReversionAgent.rsi_factor, momentum_factor, volume_spike_factor.
        """
        self.trade_horizon = trade_horizon
        if mean_reversion_window is None:
            self.mean_reversion_window = 5 if trade_horizon == "day" else 20
        else:
            self.mean_reversion_window = mean_reversion_window
        self.z_score_threshold = z_score_threshold
        self.use_volatility_adjustment = use_volatility_adjustment
        self.custom_factors = custom_factors or {}
        logger.info(
            {
                "message": f"ReversionAgent initialized with trade_horizon={trade_horizon}, mean_reversion_window={self.mean_reversion_window}, z_score_threshold={z_score_threshold}, use_volatility_adjustment={use_volatility_adjustment}, custom_factors={list(self.custom_factors.keys())}"
            }
        )

    @staticmethod
    def rsi_factor(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index) for the last value."""
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else None

    @staticmethod
    def momentum_factor(df: pd.DataFrame, window: int = 10) -> bool:
        """Detect positive momentum (last close > mean of window). Returns True if momentum is positive."""
        if len(df['Close']) < window + 1:
            return False
        if df['Close'].iloc[-1] > df['Close'].iloc[-window-1:-1].mean():
            return True
        return False

    @staticmethod
    def volume_spike_factor(df: pd.DataFrame, window: int = 20, spike_ratio: float = 2.0) -> bool:
        """Detect volume spike (last volume > spike_ratio * mean of window). Returns True if spike detected."""
        if 'Volume' not in df or len(df['Volume']) < window + 1:
            return False
        mean_vol = df['Volume'].iloc[-window-1:-1].mean()
        if mean_vol > 0 and df['Volume'].iloc[-1] > spike_ratio * mean_vol:
            return True
        return False

    @staticmethod
    def macd_factor(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Calculate MACD (Moving Average Convergence Divergence) histogram value for the last bar."""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return float(macd_hist.iloc[-1]) if not macd_hist.empty else None

    @staticmethod
    def bollinger_bands_factor(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> str:
        """Detect if price is above upper band, below lower, or within Bollinger Bands. Returns 'above', 'below', or 'within'."""
        if len(df['Close']) < window:
            return 'within'
        rolling_mean = df['Close'].rolling(window).mean()
        rolling_std = df['Close'].rolling(window).std()
        upper_band = rolling_mean + num_std * rolling_std
        lower_band = rolling_mean - num_std * rolling_std
        price = df['Close'].iloc[-1]
        if price > upper_band.iloc[-1]:
            return 'above'
        elif price < lower_band.iloc[-1]:
            return 'below'
        else:
            return 'within'

=======
    """Agent that identifies mean-reversion opportunities."""

    def __init__(self, mean_reversion_window: int = 20, z_score_threshold: float = 2.0):
        """Initialize the ReversionAgent.

        Args:
            mean_reversion_window (int): Lookback period for mean-reversion calculation.
            z_score_threshold (float): Threshold for z-score calculation.
        """
        self.mean_reversion_window = mean_reversion_window
        self.z_score_threshold = z_score_threshold
        logger.info(
            {
                "message": f"ReversionAgent initialized with mean_reversion_window={mean_reversion_window} and z_score_threshold={z_score_threshold}"
            }
        )

>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and potentially modify a trading signal.
        
        Args:
            signal (Dict[str, Any]): Trading signal to process.
        
        Returns:
            Dict[str, Any]: Processed trading signal with potential modifications.
        """
        # Default implementation: return signal as-is
        logger.info({"message": f"Processing signal: {signal}"})
        return signal

    def process(self, data: Dict) -> Dict:
<<<<<<< HEAD
        """Process market data to identify mean-reversion opportunities for swing or day trading.
=======
        """Process market data to identify mean-reversion opportunities.
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b

        Args:
            data (Dict): Market observation with 'stock_data'.

        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        """
<<<<<<< HEAD
        logger.info({"message": f"Processing data for ReversionAgent (horizon={self.trade_horizon})"})
=======
        logger.info({"message": "Processing data for ReversionAgent"})
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b
        try:
            stock_data = pd.DataFrame(data["stock_data"])
            if stock_data.empty:
                logger.warning({"message": "No stock data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            prices = stock_data["Close"]
            if len(prices) < self.mean_reversion_window:
<<<<<<< HEAD
                logger.warning({
                    "message": f"Insufficient data: {len(prices)} < {self.mean_reversion_window}"
                })
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # --- Volatility Adjustment ---
            if self.use_volatility_adjustment:
                returns = prices.pct_change().dropna()
                recent_vol = returns[-self.mean_reversion_window:].std() if len(returns) >= self.mean_reversion_window else returns.std()
                # ATR or stddev can be used; here we use stddev of returns
                window_adj = int(self.mean_reversion_window * (1 + recent_vol*5))
                threshold_adj = 0.01 + recent_vol if self.trade_horizon == "day" else 0.05 + recent_vol
                window = max(3, min(window_adj, len(prices)))
                threshold = min(0.1, threshold_adj)
            else:
                window = self.mean_reversion_window
                threshold = 0.01 if self.trade_horizon == "day" else 0.05
            mean_price = prices[-window:].mean()
            current_price = prices.iloc[-1]
            deviation = (current_price - mean_price) / mean_price

            signal_type = "day_trade" if self.trade_horizon == "day" else "swing_trade"

            # --- Multi-factor logic ---
            factors = {}
            for name, func in self.custom_factors.items():
                try:
                    factors[name] = func(stock_data)
                except Exception as e:
                    factors[name] = None
                    logger.warning({"message": f"Custom factor {name} failed: {e}"})

            # Example: require RSI < 30 for buy, > 70 for sell if RSI is present
            rsi = factors.get("rsi")
            volume_spike = factors.get("volume_spike")
            momentum = factors.get("momentum")
            action = "hold"
            confidence = 0.0
            if deviation > threshold:
                if rsi is not None and rsi > 70:
                    action = "sell"
                elif rsi is None:
                    action = "sell"
                confidence = deviation / threshold
            elif deviation < -threshold:
                if rsi is not None and rsi < 30:
                    action = "buy"
                elif rsi is None:
                    action = "buy"
                confidence = abs(deviation) / threshold
            # Optionally, volume spike or momentum can further boost confidence
            if action != "hold":
                if volume_spike:
                    confidence = min(confidence + 0.1, 1.0)
                if momentum:
                    confidence = min(confidence + 0.1, 1.0)
            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "deviation": deviation,
                    "mean_price": mean_price,
                    "signal_type": signal_type,
                    "trade_horizon": self.trade_horizon,
                    "threshold": threshold,
                    "window": window,
                    "volatility": float(recent_vol) if self.use_volatility_adjustment else None,
                    **factors
                },
=======
                logger.warning(
                    {
                        "message": f"Insufficient data: {len(prices)} < {self.mean_reversion_window}"
                    }
                )
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            mean_price = prices[-self.mean_reversion_window :].mean()
            current_price = prices.iloc[-1]
            deviation = (current_price - mean_price) / mean_price

            # Detect mean-reversion opportunity
            if deviation > 0.05:
                action = "sell"  # Overbought
                confidence = deviation
            elif deviation < -0.05:
                action = "buy"  # Oversold
                confidence = abs(deviation)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"deviation": deviation, "mean_price": mean_price},
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b
            }
            logger.info({"message": f"ReversionAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"ReversionAgent processing failed: {str(e)}"})
<<<<<<< HEAD
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e), "signal_type": self.trade_horizon}}
=======
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b

    def adapt(self, new_data: pd.DataFrame):
        """Adapt the agent to new market data (placeholder for learning).

        Args:
            new_data (pd.DataFrame): New market data.
        """
        logger.info({"message": "ReversionAgent adapting to new data"})
        try:
<<<<<<< HEAD
            # Example: Could update volatility baseline or factor weights
            pass
        except Exception as e:
            logger.error({"message": f"ReversionAgent adaptation failed: {str(e)}"})

    def backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest the agent on historical data.
        Args:
            historical_data (pd.DataFrame): DataFrame with 'Close' and any required columns.
        Returns:
            Dict[str, Any]: Results summary (PnL, win rate, etc.)
        """
        logger.info({"message": "Backtesting ReversionAgent..."})
        results = []
        for i in range(self.mean_reversion_window, len(historical_data)):
            window_data = historical_data.iloc[:i+1]
            signal = self.process({"stock_data": window_data})
            results.append(signal)
        # Simple PnL calculation: +1 for correct direction, -1 for wrong, 0 for hold
        pnl = 0
        trades = 0
        wins = 0
        for i, sig in enumerate(results[:-1]):
            next_price = historical_data["Close"].iloc[i+1]
            curr_price = historical_data["Close"].iloc[i]
            if sig["action"] == "buy":
                trades += 1
                if next_price > curr_price:
                    pnl += 1
                    wins += 1
                else:
                    pnl -= 1
            elif sig["action"] == "sell":
                trades += 1
                if next_price < curr_price:
                    pnl += 1
                    wins += 1
                else:
                    pnl -= 1
        win_rate = wins / trades if trades > 0 else 0.0
        logger.info({"message": f"Backtest complete. Trades: {trades}, Win rate: {win_rate:.2f}, PnL: {pnl}"})
        return {"trades": trades, "win_rate": win_rate, "pnl": pnl}
=======
            # Placeholder: Adjust window based on volatility
            pass
        except Exception as e:
            logger.error({"message": f"ReversionAgent adaptation failed: {str(e)}"})
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b
