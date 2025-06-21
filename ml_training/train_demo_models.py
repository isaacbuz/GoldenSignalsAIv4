#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI V3 - Demo Model Training
Trains models with 2 years of data for demonstration
Handles rate limits and works with available data
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
import time
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

class DemoDataLoader:
    """Load demo data with rate limit handling"""
    
    def __init__(self):
        # Use fewer symbols for demo
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
        
        # Use 2 years of data instead of 20
        self.start_date = datetime.now() - timedelta(days=365*2)
        self.end_date = datetime.now()
        
    def fetch_with_retry(self, symbol: str, retries: int = 3) -> pd.DataFrame:
        """Fetch data with retry logic for rate limits"""
        for attempt in range(retries):
            try:
                logger.info(f"Fetching data for {symbol} (attempt {attempt + 1})...")
                ticker = yf.Ticker(symbol)
                
                # Use shorter period to reduce API load
                data = ticker.history(period="2y", interval='1d')
                
                if not data.empty:
                    logger.info(f"✅ Successfully fetched {len(data)} days for {symbol}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")
        return pd.DataFrame()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential technical indicators"""
        df = data.copy()
        
        # Basic calculations
        df['Returns'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume ratio
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        
        # Price patterns
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Target variables
        df['Future_Return_1d'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Future_Return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['Future_Direction'] = (df['Future_Return_5d'] > 0).astype(int)
        
        # Signal classification
        df['Signal_Class'] = pd.cut(
            df['Future_Return_5d'],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=[0, 1, 2]  # Bear, Neutral, Bull
        )
        
        # Add symbol column
        df['Symbol'] = data.index.name if hasattr(data.index, 'name') else 'Unknown'
        
        return df.dropna()
    
    def load_all_data(self) -> pd.DataFrame:
        """Load data for all symbols with rate limit handling"""
        logger.info(f"🔄 Loading 2 years of demo data for {len(self.symbols)} symbols...")
        all_data = []
        
        for i, symbol in enumerate(self.symbols):
            # Add delay between requests to avoid rate limits
            if i > 0:
                time.sleep(2)
            
            data = self.fetch_with_retry(symbol)
            if not data.empty:
                # Calculate indicators
                processed_data = self.calculate_indicators(data)
                if not processed_data.empty:
                    processed_data['Symbol'] = symbol
                    all_data.append(processed_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Loaded {len(combined_data):,} total records")
            return combined_data
        else:
            logger.error("❌ No data loaded")
            return pd.DataFrame()

class DemoModelTrainer:
    """Train demo models with available data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
        # Create model directory
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets"""
        
        # Feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
            'Volume_Ratio', 'Volatility', 'High_Low_Pct'
        ]
        
        # Get available features
        available_features = [col for col in feature_columns if col in data.columns]
        X = data[available_features].copy()
        
        # Targets
        targets = {
            'price_return': data['Future_Return_5d'],
            'direction': data['Future_Direction'],
            'signal_class': data['Signal_Class']
        }
        
        return X, targets
    
    def train_forecast_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train price forecast model"""
        logger.info("🌲 Training Forecast Model...")
        
        # Remove NaN values
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            logger.warning("Not enough data for training forecast model")
            return {}
        
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
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"📊 Forecast Model - MSE: {mse:.6f}, R²: {r2:.4f}")
        
        # Save
        self.models['forecast'] = model
        self.scalers['forecast'] = scaler
        
        joblib.dump(model, self.model_dir / 'forecast_model.pkl')
        joblib.dump(scaler, self.model_dir / 'forecast_scaler.pkl')
        
        return {'mse': mse, 'r2': r2}
    
    def train_signal_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train signal classifier"""
        logger.info("🎯 Training Signal Classifier...")
        
        # Clean data
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask].astype(int)
        
        if len(X_clean) < 100:
            logger.warning("Not enough data for training signal classifier")
            return {}
        
        # Split data
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
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"📊 Signal Classifier - Accuracy: {accuracy:.4f}")
        
        # Save
        self.models['signal_classifier'] = model
        self.scalers['signal_classifier'] = scaler
        
        joblib.dump(model, self.model_dir / 'signal_classifier.pkl')
        joblib.dump(scaler, self.model_dir / 'signal_classifier_scaler.pkl')
        
        return {'accuracy': accuracy}
    
    def train_risk_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train risk assessment model"""
        logger.info("⚠️ Training Risk Model...")
        
        # Use volatility as risk proxy
        risk_values = X['Volatility'].copy()
        
        # Clean data
        mask = ~risk_values.isna()
        X_clean = X[mask]
        y_clean = risk_values[mask]
        
        if len(X_clean) < 100:
            logger.warning("Not enough data for training risk model")
            return {}
        
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
        
        logger.info(f"📊 Risk Model - MSE: {mse:.6f}")
        
        # Save
        self.models['risk'] = model
        self.scalers['risk'] = scaler
        
        joblib.dump(model, self.model_dir / 'risk_model.pkl')
        joblib.dump(scaler, self.model_dir / 'risk_scaler.pkl')
        
        return {'mse': mse}
    
    def save_training_summary(self):
        """Save training summary"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'metrics': self.metrics,
            'data_info': {
                'symbols': DemoDataLoader().symbols,
                'period': '2 years'
            }
        }
        
        with open(self.model_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Training summary saved")

def main():
    """Main demo training script"""
    print("="*60)
    print("🚀 GoldenSignalsAI V3 - Demo Model Training")
    print("📅 Training with 2 years of data (rate-limit friendly)")
    print("="*60)
    
    # Load data
    data_loader = DemoDataLoader()
    data = data_loader.load_all_data()
    
    if data.empty:
        logger.error("No data available for training")
        return
    
    # Train models
    trainer = DemoModelTrainer()
    X, targets = trainer.prepare_features(data)
    
    logger.info(f"📊 Training with {len(X):,} samples")
    
    # Train each model
    trainer.metrics['forecast'] = trainer.train_forecast_model(X, targets['price_return'])
    trainer.metrics['signal_classifier'] = trainer.train_signal_classifier(X, targets['signal_class'])
    trainer.metrics['risk'] = trainer.train_risk_model(X, targets['price_return'])
    
    # Save summary
    trainer.save_training_summary()
    
    print("\n" + "="*60)
    print("✅ Demo training completed successfully!")
    print(f"📁 Models saved to: {trainer.model_dir}")
    print("="*60)

if __name__ == "__main__":
    main() 