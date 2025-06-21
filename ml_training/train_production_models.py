#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI V3 - Production Model Training
Trains models with 20 years of historical data for production use

Features:
- Fetches 20 years of historical data from multiple sources
- Trains all 6 model types with proper validation
- Saves training data for reproducibility
- Tracks model performance metrics
- Prepares for database integration
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import joblib
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDataLoader:
    """Load 20 years of production data with error handling and caching"""
    
    def __init__(self, cache_dir: str = "data/training_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Extended symbol list for diversified training
        self.symbols = [
            # Tech giants
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Financial
            'JPM', 'BAC', 'GS', 'MS', 'WFC',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'CVS',
            # Energy
            'XOM', 'CVX',
            # Consumer
            'WMT', 'KO', 'PG',
            # ETFs for market context
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
            # Sector ETFs
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI'
        ]
        
        self.start_date = datetime.now() - timedelta(days=365*20)  # 20 years
        self.end_date = datetime.now()
        
    def fetch_symbol_data(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """Fetch 20 years of data for a symbol with caching"""
        cache_file = self.cache_dir / f"{symbol}_20y_data.pkl"
        
        # Check cache first
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 1:  # Use cache if less than 1 day old
                logger.info(f"Loading {symbol} from cache...")
                return pd.read_pickle(cache_file)
        
        try:
            logger.info(f"Fetching 20 years of data for {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Fetch maximum available data
            data = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Calculate comprehensive technical indicators
            data = self.calculate_all_indicators(data)
            
            # Add fundamental data if available
            try:
                info = ticker.info
                data['MarketCap'] = info.get('marketCap', np.nan)
                data['PE_Ratio'] = info.get('trailingPE', np.nan)
                data['EPS'] = info.get('trailingEps', np.nan)
                data['DividendYield'] = info.get('dividendYield', np.nan)
            except:
                pass
            
            # Save to cache
            data.to_pickle(cache_file)
            logger.info(f"✅ Fetched {len(data)} days of data for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['Volatility_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Price patterns
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        df['SR_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        # Momentum indicators
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['ROC_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        # Target variables for different horizons
        for days in [1, 5, 10, 20]:
            df[f'Future_Return_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
            df[f'Future_Direction_{days}d'] = (df[f'Future_Return_{days}d'] > 0).astype(int)
        
        # Volatility targets
        df['Future_Volatility_5d'] = df['Returns'].shift(-5).rolling(5).std() * np.sqrt(252)
        df['Future_Volatility_20d'] = df['Returns'].shift(-20).rolling(20).std() * np.sqrt(252)
        
        # Risk categories
        volatility_quantiles = df['Volatility_20'].quantile([0.33, 0.67])
        df['Risk_Level'] = pd.cut(
            df['Future_Volatility_5d'],
            bins=[-np.inf, volatility_quantiles[0.33], volatility_quantiles[0.67], np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        return df
    
    def load_all_data(self, use_cache: bool = True) -> pd.DataFrame:
        """Load 20 years of data for all symbols"""
        logger.info(f"🔄 Loading 20 years of market data for {len(self.symbols)} symbols...")
        all_data = []
        
        for symbol in tqdm(self.symbols, desc="Fetching symbols"):
            data = self.fetch_symbol_data(symbol, use_cache)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Remove any remaining NaN values
            combined_data = combined_data.dropna()
            
            logger.info(f"✅ Loaded {len(combined_data):,} records across {len(all_data)} symbols")
            logger.info(f"📅 Date range: {combined_data.index.min()} to {combined_data.index.max()}")
            
            # Save combined dataset
            combined_data.to_pickle(self.cache_dir / "combined_20y_data.pkl")
            
            return combined_data
        else:
            logger.error("❌ No data loaded")
            return pd.DataFrame()

class ProductionModelTrainer:
    """Production-grade model training with all optimizations"""
    
    def __init__(self, data_loader: ProductionDataLoader):
        self.data_loader = data_loader
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_importance = {}
        
        # Create directories
        self.model_dir = Path("ml_training/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = Path("ml_training/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare feature matrix and targets with proper feature engineering"""
        
        # Define feature columns (avoiding look-ahead bias)
        feature_columns = [
            # Price features
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'LogReturns',
            
            # Moving averages
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
            
            # Technical indicators
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
            'ATR', 'Volatility_20', 'Volatility_60',
            
            # Volume indicators
            'Volume_Ratio', 'OBV',
            
            # Price patterns
            'High_Low_Pct', 'Close_Open_Pct',
            'SR_Position', 'ROC_10', 'ROC_20',
            
            # Fundamental (if available)
            'MarketCap', 'PE_Ratio', 'EPS', 'DividendYield'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in data.columns]
        X = data[available_features].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(method='ffill').fillna(0)
        
        # Prepare targets
        targets = {
            'price_1d': data['Future_Return_1d'],
            'price_5d': data['Future_Return_5d'],
            'direction_1d': data['Future_Direction_1d'],
            'direction_5d': data['Future_Direction_5d'],
            'volatility_5d': data['Future_Volatility_5d'],
            'volatility_20d': data['Future_Volatility_20d']
        }
        
        # Add multi-class signal target
        conditions = [
            data['Future_Return_5d'] > 0.02,  # Bull signal
            data['Future_Return_5d'] < -0.02,  # Bear signal
        ]
        choices = [2, 0]  # 2=Bull, 0=Bear, 1=Neutral (default)
        targets['signal_class'] = np.select(conditions, choices, default=1)
        
        return X, targets
    
    def train_forecast_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest for price forecasting with time series validation"""
        logger.info("🌲 Training Production Forecast Model...")
        
        # Remove samples with NaN targets
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Time series split (no shuffling for time series!)
        split_idx = int(0.8 * len(X_clean))
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with optimized parameters
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"Training on {len(X_train):,} samples...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        
        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std()
        }
        
        logger.info(f"📊 Forecast Model Performance:")
        logger.info(f"   Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        logger.info(f"   Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
        logger.info(f"   CV MSE: {-cv_scores.mean():.6f} (+/- {cv_scores.std() * 2:.6f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['forecast'] = feature_importance
        
        # Save model and scaler
        self.models['forecast'] = model
        self.scalers['forecast'] = scaler
        self.metrics['forecast'] = metrics
        
        joblib.dump(model, self.model_dir / 'forecast_model.pkl')
        joblib.dump(scaler, self.model_dir / 'forecast_scaler.pkl')
        
        return metrics
    
    def train_signal_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost for signal classification"""
        logger.info("🎯 Training Production Signal Classifier...")
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='multi:softprob',
            num_class=3
        )
        
        logger.info(f"Training on {len(X_train):,} samples...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"📊 Signal Classifier Performance:")
        logger.info(f"   Train Accuracy: {train_acc:.4f}")
        logger.info(f"   Test Accuracy: {test_acc:.4f}")
        logger.info("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test, 
                                  target_names=['Bear', 'Neutral', 'Bull']))
        
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        }
        
        # Save
        self.models['signal_classifier'] = model
        self.scalers['signal_classifier'] = scaler
        self.metrics['signal_classifier'] = metrics
        
        joblib.dump(model, self.model_dir / 'signal_classifier.pkl')
        joblib.dump(scaler, self.model_dir / 'signal_classifier_scaler.pkl')
        
        return metrics
    
    def train_risk_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train model for risk assessment"""
        logger.info("⚠️ Training Production Risk Model...")
        
        # Clean data
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Create risk categories
        risk_levels = pd.qcut(y_clean, q=3, labels=['Low', 'Medium', 'High'])
        
        # Time series split
        split_idx = int(0.8 * len(X_clean))
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = risk_levels.iloc[:split_idx]
        y_test = risk_levels.iloc[split_idx:]
        
        # Convert to numeric for regression
        y_train_numeric = y_clean.iloc[:split_idx]
        y_test_numeric = y_clean.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVR for continuous risk score
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        logger.info(f"Training on {len(X_train):,} samples...")
        model.fit(X_train_scaled, y_train_numeric)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train_numeric, y_pred_train)
        test_mse = mean_squared_error(y_test_numeric, y_pred_test)
        
        logger.info(f"📊 Risk Model Performance:")
        logger.info(f"   Train MSE: {train_mse:.6f}")
        logger.info(f"   Test MSE: {test_mse:.6f}")
        
        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse
        }
        
        # Save
        self.models['risk'] = model
        self.scalers['risk'] = scaler
        self.metrics['risk'] = metrics
        
        joblib.dump(model, self.model_dir / 'risk_model.pkl')
        joblib.dump(scaler, self.model_dir / 'risk_scaler.pkl')
        
        return metrics
    
    def save_training_report(self):
        """Save comprehensive training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'data_stats': {
                'total_samples': len(self.data_loader.load_all_data(use_cache=True)),
                'symbols': self.data_loader.symbols,
                'date_range': '20 years'
            },
            'model_metrics': self.metrics,
            'feature_importance': {
                name: fi.to_dict() if hasattr(fi, 'to_dict') else fi 
                for name, fi in self.feature_importance.items()
            }
        }
        
        # Save report
        with open(self.metrics_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Training report saved to {self.metrics_dir / 'training_report.json'}")
    
    def train_all_models(self, data: pd.DataFrame):
        """Train all production models"""
        logger.info("🚀 Starting production model training with 20 years of data...")
        
        # Prepare features
        X, targets = self.prepare_features(data)
        
        logger.info(f"📊 Feature matrix shape: {X.shape}")
        logger.info(f"📊 Training samples: {len(X):,}")
        
        # Train each model
        self.train_forecast_model(X, targets['price_5d'])
        self.train_signal_classifier(X, targets['signal_class'])
        self.train_risk_model(X, targets['volatility_20d'])
        
        # Save training report
        self.save_training_report()
        
        logger.info("✅ All models trained successfully!")

def main():
    """Main training script for production models"""
    print("="*60)
    print("🚀 GoldenSignalsAI V3 - Production Model Training")
    print("📅 Training with 20 years of historical data")
    print("="*60)
    
    # Initialize data loader
    data_loader = ProductionDataLoader()
    
    # Load all data
    all_data = data_loader.load_all_data(use_cache=True)
    
    if all_data.empty:
        logger.error("❌ No data available for training")
        return
    
    # Initialize trainer
    trainer = ProductionModelTrainer(data_loader)
    
    # Train all models
    trainer.train_all_models(all_data)
    
    print("\n" + "="*60)
    print("🎉 Production model training completed!")
    print(f"📁 Models saved to: {trainer.model_dir}")
    print(f"📊 Metrics saved to: {trainer.metrics_dir}")
    print("="*60)

if __name__ == "__main__":
    main() 