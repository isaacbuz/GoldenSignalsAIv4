import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = "/Users/isaacbuz/Documents/Projects/GoldenSignalsAI"

dirs = [
    "domain/trading/entities",
    "domain/trading/strategies",
    "domain/trading/agents",
    "domain/models",
    "domain/tests",
    "domain/risk_management",
    "domain/backtesting",
    "domain/portfolio",
    "domain/analytics"
]

files = {
    "domain/trading/entities/__init__.py": "# domain/trading/entities/__init__.py\n",
    "domain/trading/entities/trade.py": """from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    symbol: str
    quantity: int
    price: float
    timestamp: datetime = datetime.now()
    action: str = "Buy"
    stop_loss: float = None
    take_profit: float = None
""",
    "domain/trading/entities/user_preferences.py": """from dataclasses import dataclass

@dataclass
class UserPreferences:
    user_id: int
    phone_number: str
    whatsapp_number: str
    x_enabled: bool
    enabled_channels: list
    price_threshold: float
""",
    "domain/trading/strategies/__init__.py": "# domain/trading/strategies/__init__.py\n",
    "domain/trading/strategies/backtest_strategy.py": """import pandas as pd

class BacktestStrategy:
    def run(self, symbol, historical_df, predictions):
        returns = pd.Series(predictions).pct_change().fillna(0)
        total_return = returns.sum()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown)
        }
""",
    "domain/trading/strategies/indicators.py": """import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def moving_average(self, window):
        return self.data['close'].rolling(window=window).mean()

    def exponential_moving_average(self, window):
        return self.data['close'].ewm(span=window, adjust=False).mean()

    def vwap(self):
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        vwap = (typical_price * self.data['volume']).cumsum() / self.data['volume'].cumsum()
        return vwap

    def bollinger_bands(self, window):
        sma = self.moving_average(window)
        std = self.data['close'].rolling(window=window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        return upper_band, sma, lower_band

    def rsi(self, window):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd(self, fast, slow, signal):
        ema_fast = self.exponential_moving_average(fast)
        ema_slow = self.exponential_moving_average(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
""",
    "domain/trading/strategies/signal_engine.py": """import pandas as pd

class SignalEngine:
    def __init__(self, data, weights=None):
        self.data = data
        self.weights = weights or {
            "ma_cross": 0.2,
            "ema_cross": 0.2,
            "vwap": 0.1,
            "bollinger": 0.2,
            "rsi": 0.15,
            "macd": 0.15
        }

    def compute_signal_score(self):
        return 0.75

    def generate_signal(self, symbol, risk_profile="balanced"):
        latest_price = self.data['close'].iloc[-1]
        return {
            "symbol": symbol,
            "action": "Buy",
            "price": latest_price,
            "confidence_score": 0.75,
            "stop_loss": latest_price * 0.98,
            "profit_target": latest_price * 1.04
        }
""",
    "domain/trading/strategies/trading_env.py": """import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, symbol):
        super(TradingEnv, self).__init__()
        self.data = data
        self.symbol = symbol
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.cash = 10000
        self.shares = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.cash = 10000
        self.shares = 0
        return self._get_observation()

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0
        done = False
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.shares = self.cash // current_price
            self.cash -= self.shares * current_price
        elif action == 2 and self.position == 1:
            self.position = 0
            profit = (current_price - self.entry_price) * self.shares
            self.cash += self.shares * current_price
            self.shares = 0
            reward = profit
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.data.iloc[self.current_step].values
""",
    "domain/trading/agents/__init__.py": "# domain/trading/agents/__init__.py\n",
    "domain/models/__init__.py": "# domain/models/__init__.py\n",
    "domain/models/factory.py": """class ModelFactory:
    def create_model(self, model_type):
        if model_type == "lstm":
            return MockLSTMModel()
        elif model_type == "xgboost":
            return MockXGBoostModel()
        elif model_type == "lightgbm":
            return MockLightGBMModel()
        elif model_type == "sentiment":
            return MockSentimentModel()
        elif model_type == "rl":
            return MockRLModel()
        raise ValueError(f"Unknown model type: {model_type}")

class MockLSTMModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return 280.0

class MockXGBoostModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return 0.05

class MockLightGBMModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return 0.04

class MockSentimentModel:
    def analyze(self, news_articles):
        return 0.1

class MockRLModel:
    def train(self, df):
        pass
""",
    "domain/tests/__init__.py": "# domain/tests/__init__.py\n",
    "domain/risk_management/__init__.py": "# domain/risk_management/__init__.py\n",
    "domain/risk_management/engine.py": """from enum import Enum
import numpy as np

class RiskLevel(Enum):
    CONSERVATIVE = 1
    BALANCED = 2
    AGGRESSIVE = 3

class RiskEngine:
    def __init__(self):
        self.risk_params = {
            RiskLevel.CONSERVATIVE: {'max_loss': 0.01, 'position_size': 0.02},
            RiskLevel.BALANCED: {'max_loss': 0.02, 'position_size': 0.05},
            RiskLevel.AGGRESSIVE: {'max_loss': 0.03, 'position_size': 0.1}
        }

    def calculate_position_size(self, portfolio_value, volatility, risk_level):
        params = self.risk_params[risk_level]
        base_size = portfolio_value * params['position_size']
        return min(base_size, base_size / (volatility * 10))

    def calculate_risk(self, trade):
        return 0.1
""",
    "domain/backtesting/__init__.py": "# domain/backtesting/__init__.py\n",
    "domain/backtesting/backtest_engine.py": """import pandas as pd

class BacktestEngine:
    def run(self, data, params):
        returns = pd.Series(data['close']).pct_change().fillna(0)
        total_return = returns.sum()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown)
        }
""",
    "domain/portfolio/__init__.py": "# domain/portfolio/__init__.py\n",
    "domain/portfolio/portfolio_manager.py": """class PortfolioManager:
    def __init__(self):
        self.positions = {}

    def add_position(self, trade):
        self.positions[trade.symbol] = trade

    def get_portfolio_value(self):
        return 10000.0
""",
    "domain/analytics/__init__.py": "# domain/analytics/__init__.py\n",
    "domain/analytics/decision_explainer.py": """class DecisionExplainer:
    def explain(self, decision):
        return {
            "symbol": decision.symbol,
            "action": decision.action.name,
            "confidence": decision.confidence,
            "rationale": decision.rationale
        }
"""
}

for directory in dirs:
    try:
        full_path = os.path.join(BASE_DIR, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")

for file_path, content in files.items():
    try:
        full_path = os.path.join(BASE_DIR, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        logger.info(f"Created file: {file_path}")
    except Exception as e:
        logger.error(f"Error creating file {file_path}: {str(e)}")

logger.info("Domain files generation complete.")