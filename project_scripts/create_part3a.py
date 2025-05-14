# create_part3a.py
# Purpose: Creates files in the domain/ directory for the GoldenSignalsAI project,
# including data models and trading logic. Incorporates improvements like decoupled Pydantic
# models for stock, signal, and options data, and enhanced regime detection for options trading strategies.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part3a():
    """Create files in domain/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating domain files in {base_dir}"})

    # Define domain directory files
    domain_files = {
        "domain/__init__.py": """# domain/__init__.py
# Purpose: Marks the domain directory as a Python subpackage, enabling imports
# for domain models and trading logic.

# Empty __init__.py to mark domain as a subpackage
""",
        "domain/models/__init__.py": """# domain/models/__init__.py
# Purpose: Marks the models directory as a Python subpackage, enabling imports
# for data models (stock, signal, options) and AI models.

# Empty __init__.py to mark models as a subpackage
""",
        "domain/models/stock.py": """# domain/models/stock.py
# Purpose: Defines a Pydantic model for validating stock data, ensuring data integrity
# for trading signals and options analysis. Split from data_models.py for modularity.

from pydantic import BaseModel, Field
import logging

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class StockData(BaseModel):
    \"\"\"Pydantic model for stock data validation.\"\"\"
    timestamp: str = Field(..., alias='date', description="Timestamp of the data point")
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")

    class Config:
        \"\"\"Pydantic configuration for alias support.\"\"\"
        allow_population_by_field_name = True

    def validate_data(self):
        \"\"\"Validate the stock data instance.
        
        Returns:
            bool: True if valid, False otherwise.
        \"\"\"
        logger.debug({"message": f"Validating stock data for {self.symbol}"})
        try:
            self.validate()
            logger.debug({"message": f"Stock data validated successfully for {self.symbol}"})
            return True
        except Exception as e:
            logger.error({"message": f"Stock data validation failed for {self.symbol}: {str(e)}"})
            return False
""",
        "domain/models/signal.py": """# domain/models/signal.py
# Purpose: Defines a Pydantic model for validating trading signals, ensuring consistency
# in signal generation for options trading. Split from data_models.py for modularity.

from pydantic import BaseModel, Field
import logging
from typing import List, Dict

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class TradingSignal(BaseModel):
    \"\"\"Pydantic model for trading signal validation.\"\"\"
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    action: str = Field(..., description="Trading action ('buy', 'sell', 'hold')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0 to 1)")
    ai_score: float = Field(..., description="AI prediction score")
    indicator_score: float = Field(..., description="Technical indicator score")
    final_score: float = Field(..., description="Final combined score")
    timestamp: str = Field(..., description="Timestamp of the signal")
    risk_profile: str = Field(..., description="User risk profile ('conservative', 'balanced', 'aggressive')")
    indicators: List[str] = Field(..., description="List of indicators used")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata (e.g., regime)")

    class Config:
        \"\"\"Pydantic configuration for validation.\"\"\"
        allow_population_by_field_name = True

    def validate_data(self):
        \"\"\"Validate the trading signal instance.
        
        Returns:
            bool: True if valid, False otherwise.
        \"\"\"
        logger.debug({"message": f"Validating trading signal for {self.symbol}"})
        try:
            if self.action not in ['buy', 'sell', 'hold']:
                raise ValueError(f"Invalid action: {self.action}")
            if self.risk_profile not in ['conservative', 'balanced', 'aggressive']:
                raise ValueError(f"Invalid risk profile: {self.risk_profile}")
            self.validate()
            logger.debug({"message": f"Trading signal validated successfully for {self.symbol}"})
            return True
        except Exception as e:
            logger.error({"message": f"Trading signal validation failed for {self.symbol}: {str(e)}"})
            return False
""",
        "domain/models/options.py": """# domain/models/options.py
# Purpose: Defines a Pydantic model for validating options chain data, supporting
# options trading analysis with fields for volume, open interest, and Greeks.
# Split from data_models.py for modularity and specificity.

from pydantic import BaseModel, Field
import logging

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class OptionsData(BaseModel):
    \"\"\"Pydantic model for options chain data validation.\"\"\"
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    call_volume: float = Field(..., ge=0, description="Call option trading volume")
    put_volume: float = Field(..., ge=0, description="Put option trading volume")
    call_oi: float = Field(..., ge=0, description="Call option open interest")
    put_oi: float = Field(..., ge=0, description="Put option open interest")
    strike: float = Field(..., gt=0, description="Strike price")
    call_put: str = Field(..., description="Option type ('call' or 'put')")
    iv: float = Field(..., ge=0, description="Implied volatility")
    quantity: float = Field(..., ge=0, description="Number of contracts")
    delta: float = Field(None, description="Option delta (optional)")
    gamma: float = Field(None, description="Option gamma (optional)")
    theta: float = Field(None, description="Option theta (optional)")

    class Config:
        \"\"\"Pydantic configuration for validation.\"\"\"
        allow_population_by_field_name = True

    def validate_data(self):
        \"\"\"Validate the options data instance.
        
        Returns:
            bool: True if valid, False otherwise.
        \"\"\"
        logger.debug({"message": f"Validating options data for {self.symbol}"})
        try:
            if self.call_put not in ['call', 'put']:
                raise ValueError(f"Invalid option type: {self.call_put}")
            self.validate()
            logger.debug({"message": f"Options data validated successfully for {self.symbol}"})
            return True
        except Exception as e:
            logger.error({"message": f"Options data validation failed for {self.symbol}: {str(e)}"})
            return False
""",
        "domain/models/ai_models.py": """# domain/models/ai_models.py
# Purpose: Defines mock AI model classes for use in the model factory, supporting
# various machine learning and statistical models for options trading signal generation.

import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class BaseModel:
    \"\"\"Base class for all AI models with mock implementations.\"\"\"
    def train(self, data, **kwargs):
        logger.info({"message": f"Training mock {self.__class__.__name__}"})
        # Mock training
        pass

    def predict(self, data):
        logger.info({"message": f"Predicting with mock {self.__class__.__name__}"})
        # Mock prediction: return random values
        if isinstance(data, np.ndarray):
            return np.random.randn(data.shape[0])
        elif isinstance(data, pd.DataFrame):
            return np.random.randn(len(data))
        return np.random.randn(1)

class LSTMModel(BaseModel):
    \"\"\"Mock LSTM model.\"\"\"
    def __init__(self, scaler: MinMaxScaler = None):
        self.scaler = scaler or MinMaxScaler()

class GRUModel(BaseModel):
    \"\"\"Mock GRU model.\"\"\"
    def __init__(self, scaler: MinMaxScaler = None):
        self.scaler = scaler or MinMaxScaler()

class TransformerModel(BaseModel):
    \"\"\"Mock Transformer model.\"\"\"
    def __init__(self, scaler: MinMaxScaler = None):
        self.scaler = scaler or MinMaxScaler()

class CNNModel(BaseModel):
    \"\"\"Mock CNN model.\"\"\"
    def __init__(self, scaler: MinMaxScaler = None):
        self.scaler = scaler or MinMaxScaler()

class XGBoostModel(BaseModel):
    \"\"\"Mock XGBoost model.\"\"\"
    pass

class LightGBMModel(BaseModel):
    \"\"\"Mock LightGBM model.\"\"\"
    pass

class CatBoostModel(BaseModel):
    \"\"\"Mock CatBoost model.\"\"\"
    pass

class RandomForestModel(BaseModel):
    \"\"\"Mock Random Forest model.\"\"\"
    pass

class GradientBoostingModel(BaseModel):
    \"\"\"Mock Gradient Boosting model.\"\"\"
    pass

class SVMModel(BaseModel):
    \"\"\"Mock SVM model.\"\"\"
    pass

class KNNModel(BaseModel):
    \"\"\"Mock KNN model.\"\"\"
    pass

class ARIMAModel(BaseModel):
    \"\"\"Mock ARIMA model.\"\"\"
    pass

class GARCHModel(BaseModel):
    \"\"\"Mock GARCH model.\"\"\"
    pass

class DQNModel(BaseModel):
    \"\"\"Mock DQN model.\"\"\"
    pass

class GaussianProcessModel(BaseModel):
    \"\"\"Mock Gaussian Process model.\"\"\"
    pass
""",
        "domain/trading/__init__.py": """# domain/trading/__init__.py
# Purpose: Marks the trading directory as a Python subpackage, enabling imports
# for technical indicators and regime detection.

# Empty __init__.py to mark trading as a subpackage
""",
        "domain/trading/indicators.py": """# domain/trading/indicators.py
# Purpose: Computes technical indicators (e.g., RSI, MACD) and adjusts signals based on
# market regime detection (trending, mean-reverting, volatile). Enhanced for options trading
# to support regime-adjusted strategies.

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

class Indicators:
    \"\"\"Computes technical indicators and detects market regimes.\"\"\"
    def __init__(self, stock_data: pd.DataFrame, intraday_data: pd.DataFrame = None, weekly_data: pd.DataFrame = None):
        \"\"\"Initialize with multi-timeframe data.
        
        Args:
            stock_data (pd.DataFrame): Daily stock data.
            intraday_data (pd.DataFrame, optional): Intraday data.
            weekly_data (pd.DataFrame, optional): Weekly data.
        \"\"\"
        self.stock_data = stock_data
        self.intraday_data = intraday_data
        self.weekly_data = weekly_data
        logger.info({"message": "Indicators initialized"})

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        \"\"\"Compute Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series.
            period (int): Lookback period (default: 14).
        
        Returns:
            pd.Series: RSI values.
        \"\"\"
        logger.debug({"message": f"Computing RSI with period={period}"})
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error({"message": f"Failed to compute RSI: {str(e)}"})
            return pd.Series(index=prices.index)

    def compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        \"\"\"Compute Moving Average Convergence Divergence (MACD).
        
        Args:
            prices (pd.Series): Price series.
            fast (int): Fast EMA period (default: 12).
            slow (int): Slow EMA period (default: 26).
            signal (int): Signal line period (default: 9).
        
        Returns:
            pd.Series: MACD line values.
        \"\"\"
        logger.debug({"message": f"Computing MACD with fast={fast}, slow={slow}, signal={signal}"})
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception as e:
            logger.error({"message": f"Failed to compute MACD: {str(e)}"})
            return pd.Series(index=prices.index)

    def detect_market_regime(self) -> str:
        \"\"\"Detect the current market regime based on multi-timeframe data.
        
        Returns:
            str: Market regime ('trending', 'mean_reverting', 'volatile').
        \"\"\"
        logger.info({"message": "Detecting market regime"})
        try:
            # Simplified regime detection based on volatility and trend
            prices = self.stock_data['close']
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            trend = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]

            if volatility > 0.3:
                regime = 'volatile'
            elif abs(trend) > 0.05:
                regime = 'trending'
            else:
                regime = 'mean_reverting'
            
            logger.info({"message": f"Detected market regime: {regime}"})
            return regime
        except Exception as e:
            logger.error({"message": f"Failed to detect market regime: {str(e)}"})
            return 'mean_reverting'

    def compute_regime_adjusted_signal(self, indicators: List[str]) -> Dict[str, float]:
        \"\"\"Compute regime-adjusted signals for selected indicators.
        
        Args:
            indicators (List[str]): List of indicators to compute (e.g., ['RSI', 'MACD']).
        
        Returns:
            Dict[str, float]: Dictionary of indicator signals adjusted for regime.
        \"\"\"
        logger.info({"message": f"Computing regime-adjusted signals for {indicators}"})
        try:
            regime = self.detect_market_regime()
            signals = {}
            prices = self.stock_data['close']

            for indicator in indicators:
                if indicator == 'RSI':
                    rsi = self.compute_rsi(prices)
                    signal = -1 if rsi.iloc[-1] > 70 else 1 if rsi.iloc[-1] < 30 else 0
                elif indicator == 'MACD':
                    macd = self.compute_macd(prices)
                    signal = 1 if macd.iloc[-1] > 0 else -1 if macd.iloc[-1] < 0 else 0
                else:
                    signal = 0
                
                # Adjust signal based on regime for options trading
                if regime == 'trending':
                    signals[indicator] = signal * 1.2  # Amplify in trending markets (directional options)
                elif regime == 'mean_reverting':
                    signals[indicator] = signal * 1.5  # Amplify in mean-reverting markets (straddles)
                else:  # volatile
                    signals[indicator] = signal * 0.8  # Reduce in volatile markets (hedging)

            logger.info({"message": f"Computed signals: {signals}"})
            return signals
        except Exception as e:
            logger.error({"message": f"Failed to compute regime-adjusted signals: {str(e)}"})
            return {}
""",
        "domain/trading/regime_detector.py": """# domain/trading/regime_detector.py
# Purpose: Detects market regimes (trending, mean-reverting, volatile) based on price
# and volatility patterns. Used by agents to adjust trading strategies, particularly
# for options trading where regime impacts strategy selection.

import pandas as pd
import numpy as np
import logging

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class RegimeDetector:
    \"\"\"Detects market regimes based on price and volatility data.\"\"\"
    def __init__(self, window: int = 20):
        \"\"\"Initialize with a lookback window.
        
        Args:
            window (int): Lookback period for regime detection (default: 20).
        \"\"\"
        self.window = window
        logger.info({"message": f"RegimeDetector initialized with window={window}"})

    def detect(self, prices: pd.Series) -> str:
        \"\"\"Detect the market regime based on price data.
        
        Args:
            prices (pd.Series): Price series.
        
        Returns:
            str: Market regime ('trending', 'mean_reverting', 'volatile').
        \"\"\"
        logger.info({"message": f"Detecting regime for price series with length={len(prices)}"})
        try:
            if len(prices) < self.window:
                logger.warning({"message": f"Insufficient data for regime detection: {len(prices)} < {self.window}"})
                return "mean_reverting"

            # Calculate returns and volatility
            returns = prices.pct_change().dropna()
            volatility = returns[-self.window:].std() * np.sqrt(252)
            # Calculate trend strength
            trend = (prices.iloc[-1] - prices.iloc[-self.window]) / prices.iloc[-self.window]

            # Determine regime for options trading
            if volatility > 0.3:
                regime = 'volatile'  # Suitable for volatility-based options strategies
            elif abs(trend) > 0.05:
                regime = 'trending'  # Suitable for directional options plays
            else:
                regime = 'mean_reverting'  # Suitable for mean-reversion options strategies

            logger.info({
                "message": f"Detected regime: {regime}",
                "volatility": volatility,
                "trend": trend
            })
            return regime
        except Exception as e:
            logger.error({"message": f"Failed to detect regime: {str(e)}"})
            return "mean_reverting"
""",
    }

    # Write domain directory files
    for file_path, content in domain_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 3a: domain/ created successfully")


if __name__ == "__main__":
    create_part3a()
