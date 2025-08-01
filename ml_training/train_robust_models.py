#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI V3 - Robust Model Training
Handles multiple data sources and can train with synthetic data if needed
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustDataLoader:
    """Load data from multiple sources with fallback options"""

    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ']
        self.data_sources = []

    def try_yfinance(self, symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
        """Try to fetch data using yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Try different approaches
            end = datetime.now()
            start = end - timedelta(days=days)

            # Approach 1: Direct download
            data = yf.download(symbol, start=start, end=end, progress=False)
            if not data.empty:
                # Handle multi-level columns from yf.download
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                logger.info(f"✅ yfinance: Got {len(data)} records for {symbol}")
                return data

        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")

        return None

    def try_pandas_datareader(self, symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
        """Try pandas_datareader as alternative"""
        try:
            import pandas_datareader as pdr
            end = datetime.now()
            start = end - timedelta(days=days)

            # Try different sources
            for source in ['yahoo', 'iex', 'av-daily']:
                try:
                    data = pdr.DataReader(symbol, source, start, end)
                    if not data.empty:
                        logger.info(f"✅ pandas_datareader ({source}): Got {len(data)} records for {symbol}")
                        return data
                except:
                    continue

        except Exception as e:
            logger.warning(f"pandas_datareader failed for {symbol}: {e}")

        return None

    def generate_synthetic_data(self, symbol: str, days: int = 730) -> pd.DataFrame:
        """Generate realistic synthetic market data for training"""
        logger.info(f"🔧 Generating synthetic data for {symbol} ({days} days)")

        # Create date range
        end = datetime.now()
        start = end - timedelta(days=days)
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days only

        # Generate realistic price movements
        np.random.seed(42)  # For reproducibility

        # Base price for different symbols
        base_prices = {
            'AAPL': 150, 'MSFT': 300, 'GOOGL': 120,
            'AMZN': 130, 'SPY': 450, 'QQQ': 380
        }
        base_price = base_prices.get(symbol, 100)

        # Generate returns with realistic properties
        # Daily returns: normal distribution with slight positive drift
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily drift, 2% volatility

        # Add some autocorrelation (momentum)
        for i in range(1, len(daily_returns)):
            daily_returns[i] = 0.1 * daily_returns[i-1] + 0.9 * daily_returns[i]

        # Generate prices
        prices = base_price * np.exp(np.cumsum(daily_returns))

        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices

        # High/Low based on daily volatility
        daily_range = np.abs(np.random.normal(0, 0.01, len(dates)))
        data['High'] = data['Close'] * (1 + daily_range)
        data['Low'] = data['Close'] * (1 - daily_range * 0.8)  # Asymmetric

        # Open price
        data['Open'] = data['Close'].shift(1)
        data['Open'].iloc[0] = base_price

        # Volume with some patterns
        base_volume = 50_000_000 if symbol in ['SPY', 'QQQ'] else 20_000_000
        volume_noise = np.random.lognormal(0, 0.5, len(dates))
        data['Volume'] = (base_volume * volume_noise).astype(int)

        # Add some realistic patterns
        # 1. Higher volume on big moves
        big_moves = np.abs(daily_returns) > 0.02
        data.loc[big_moves, 'Volume'] *= 2

        # 2. Trend in volume over time
        volume_trend = np.linspace(0.8, 1.2, len(dates))
        data['Volume'] = (data['Volume'] * volume_trend).astype(int)

        # Ensure OHLC relationships are valid
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

        return data

    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()

        # Basic calculations
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std).values
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std).values
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['ATR'] = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift()),
            np.abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1).rolling(14).mean()

        # Price patterns
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']

        # Support and Resistance
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        df['SR_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])

        # Target variables
        for days in [1, 5, 10]:
            df[f'Future_Return_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
            df[f'Future_Direction_{days}d'] = (df[f'Future_Return_{days}d'] > 0).astype(int)

        # Signal classification (Bear: 0, Neutral: 1, Bull: 2)
        df['Signal_Class'] = pd.cut(
            df['Future_Return_5d'],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=[0, 1, 2]
        )

        # Risk level based on future volatility
        df['Future_Volatility'] = df['Returns'].shift(-20).rolling(20).std() * np.sqrt(252)

        # Add symbol
        df['Symbol'] = symbol

        return df.dropna()

    def load_data(self, use_synthetic_if_needed: bool = True) -> pd.DataFrame:
        """Load data with multiple fallback options"""
        logger.info(f"🔄 Loading market data for {len(self.symbols)} symbols...")
        all_data = []
        data_sources = {}

        for symbol in self.symbols:
            logger.info(f"\n📊 Processing {symbol}...")

            # Try real data sources first
            data = None
            source = None

            # Try yfinance
            data = self.try_yfinance(symbol)
            if data is not None:
                source = "yfinance"

            # Try pandas_datareader if yfinance failed
            if data is None:
                data = self.try_pandas_datareader(symbol)
                if data is not None:
                    source = "pandas_datareader"

            # Use synthetic data as last resort
            if data is None and use_synthetic_if_needed:
                data = self.generate_synthetic_data(symbol)
                source = "synthetic"

            if data is not None:
                # Calculate indicators
                processed_data = self.calculate_indicators(data, symbol)
                if not processed_data.empty:
                    all_data.append(processed_data)
                    data_sources[symbol] = source
                    logger.info(f"✅ Loaded {len(processed_data)} records from {source}")
            else:
                logger.error(f"❌ Failed to load any data for {symbol}")

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n✅ Total records loaded: {len(combined_data):,}")
            logger.info(f"📊 Data sources: {data_sources}")
            return combined_data
        else:
            logger.error("❌ No data loaded for any symbol")
            return pd.DataFrame()

class RobustModelTrainer:
    """Train models with whatever data is available"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_importance = {}

        # Create directories
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)

    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets"""

        # Core feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'LogReturns',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volume_Ratio', 'OBV',
            'Volatility_20', 'ATR',
            'High_Low_Pct', 'Close_Open_Pct',
            'SR_Position'
        ]

        # Get available features
        available_features = [col for col in feature_columns if col in data.columns]
        X = data[available_features].copy()

        # Targets
        targets = {
            'price_return': data['Future_Return_5d'],
            'direction': data['Future_Direction_5d'],
            'signal_class': data['Signal_Class'],
            'volatility': data['Future_Volatility']
        }

        logger.info(f"📊 Features: {len(available_features)}, Samples: {len(X)}")

        return X, targets

    def train_all_models(self, data: pd.DataFrame):
        """Train all model types"""
        logger.info("\n🚀 Starting model training...")

        # Prepare features
        X, targets = self.prepare_features(data)

        if len(X) < 100:
            logger.error("Not enough data for training (need at least 100 samples)")
            return

        # 1. Price Forecast Model
        logger.info("\n🌲 Training Random Forest Price Forecast Model...")
        self.train_forecast_model(X, targets['price_return'], 'forecast')

        # 2. Signal Classifier
        logger.info("\n🎯 Training Gradient Boosting Signal Classifier...")
        self.train_signal_classifier(X, targets['signal_class'], 'signal_classifier')

        # 3. Risk Model
        logger.info("\n⚠️ Training SVR Risk Model...")
        self.train_risk_model(X, targets['volatility'], 'risk')

        # 4. Direction Classifier
        logger.info("\n📈 Training Direction Classifier...")
        self.train_direction_classifier(X, targets['direction'], 'direction')

        # Save summary
        self.save_training_summary()

    def train_forecast_model(self, X: pd.DataFrame, y: pd.Series, name: str):
        """Train regression model for price forecasting"""
        # Clean data
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Time series split
        split_idx = int(0.8 * len(X_clean))
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        logger.info(f"   Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        logger.info(f"   Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")

        # Feature importance
        self.feature_importance[name] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save
        self.models[name] = model
        self.scalers[name] = scaler
        self.metrics[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

        joblib.dump(model, self.model_dir / f'{name}_model.pkl')
        joblib.dump(scaler, self.model_dir / f'{name}_scaler.pkl')

    def train_signal_classifier(self, X: pd.DataFrame, y: pd.Series, name: str):
        """Train multi-class classifier"""
        # Clean data
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask].astype(int)

        # Split
        split_idx = int(0.8 * len(X_clean))
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=['Bear', 'Neutral', 'Bull']
        ))

        # Save
        self.models[name] = model
        self.scalers[name] = scaler
        self.metrics[name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        joblib.dump(model, self.model_dir / f'{name}_model.pkl')
        joblib.dump(scaler, self.model_dir / f'{name}_scaler.pkl')

    def train_risk_model(self, X: pd.DataFrame, y: pd.Series, name: str):
        """Train risk assessment model"""
        # Clean data
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Split
        split_idx = int(0.8 * len(X_clean))
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"   MSE: {mse:.6f}, R²: {r2:.4f}")

        # Save
        self.models[name] = model
        self.scalers[name] = scaler
        self.metrics[name] = {'mse': mse, 'r2': r2}

        joblib.dump(model, self.model_dir / f'{name}_model.pkl')
        joblib.dump(scaler, self.model_dir / f'{name}_scaler.pkl')

    def train_direction_classifier(self, X: pd.DataFrame, y: pd.Series, name: str):
        """Train binary direction classifier"""
        # Clean data
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask].astype(int)

        # Split
        split_idx = int(0.8 * len(X_clean))
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"   Accuracy: {accuracy:.4f}")

        # Save
        self.models[name] = model
        self.scalers[name] = scaler
        self.metrics[name] = {'accuracy': accuracy}

        joblib.dump(model, self.model_dir / f'{name}_model.pkl')
        joblib.dump(scaler, self.model_dir / f'{name}_scaler.pkl')

    def save_training_summary(self):
        """Save comprehensive training summary"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'metrics': self.metrics,
            'feature_importance': {
                name: fi.head(10).to_dict() if hasattr(fi, 'to_dict') else fi
                for name, fi in self.feature_importance.items()
            },
            'model_files': [str(f) for f in self.model_dir.glob('*.pkl')]
        }

        with open(self.metrics_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\n📄 Training summary saved to {self.metrics_dir / 'training_summary.json'}")

def main():
    """Main training script"""
    print("="*60)
    print("🚀 GoldenSignalsAI V3 - Robust Model Training")
    print("📊 Training with multiple data sources and fallbacks")
    print("="*60)

    # Load data
    data_loader = RobustDataLoader()
    data = data_loader.load_data(use_synthetic_if_needed=True)

    if data.empty:
        logger.error("No data available for training")
        return

    # Train models
    trainer = RobustModelTrainer()
    trainer.train_all_models(data)

    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print(f"📁 Models saved to: {trainer.model_dir}")
    print(f"📊 Metrics saved to: {trainer.metrics_dir}")
    print("\n🎯 Next steps:")
    print("1. Review training metrics in metrics/training_summary.json")
    print("2. Test models with test_models.py")
    print("3. Deploy models to production when ready")
    print("="*60)

if __name__ == "__main__":
    main()
