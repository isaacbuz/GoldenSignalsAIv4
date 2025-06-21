#!/usr/bin/env python3
"""
Advanced Backtesting System for GoldenSignalsAI
Validates ML models against historical data and improves accuracy
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedBacktestEngine:
    """Advanced backtesting engine with ML model validation"""
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
        self.signal_history = []
        
    async def backtest_with_ml_validation(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        ml_models: Optional[Dict] = None,
        signal_generator: Optional[Callable] = None
    ) -> Dict:
        """
        Run comprehensive backtest with ML model validation
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for backtest
            end_date: End date for backtest
            ml_models: Dictionary of ML models to validate
            signal_generator: Function to generate signals
            
        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting advanced backtest for {len(symbols)} symbols")
        
        results = {
            'symbol_results': {},
            'model_performance': {},
            'accuracy_metrics': {},
            'profit_metrics': {},
            'risk_metrics': {},
            'optimization_suggestions': []
        }
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")
            
            # Fetch historical data
            data = self._fetch_historical_data(symbol, start_date, end_date)
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue
                
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Run walk-forward analysis
            symbol_results = await self._walk_forward_analysis(
                symbol, data, ml_models, signal_generator
            )
            
            results['symbol_results'][symbol] = symbol_results
            
        # Calculate aggregate metrics
        results['accuracy_metrics'] = self._calculate_accuracy_metrics(results['symbol_results'])
        results['profit_metrics'] = self._calculate_profit_metrics(results['symbol_results'])
        results['risk_metrics'] = self._calculate_risk_metrics(results['symbol_results'])
        results['optimization_suggestions'] = self._generate_optimization_suggestions(results)
        
        return results
    
    def _fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical data with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
                
            # Add additional price features
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = self._calculate_macd(data['Close'])
        
        # Bollinger Bands
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # ATR (Average True Range)
        data['ATR'] = self._calculate_atr(data)
        
        # Stochastic Oscillator
        data['Stoch_K'], data['Stoch_D'] = self._calculate_stochastic(data)
        
        # Support and Resistance levels
        data['Support'], data['Resistance'] = self._calculate_support_resistance(data)
        
        return data.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate dynamic support and resistance levels"""
        support = data['Low'].rolling(window=window).min()
        resistance = data['High'].rolling(window=window).max()
        return support, resistance
    
    async def _walk_forward_analysis(
        self,
        symbol: str,
        data: pd.DataFrame,
        ml_models: Optional[Dict],
        signal_generator: Optional[Callable]
    ) -> Dict:
        """Perform walk-forward analysis"""
        
        results = {
            'predictions': [],
            'actual_movements': [],
            'profits': [],
            'accuracy_over_time': [],
            'model_scores': {}
        }
        
        # Use time series split for proper validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Generate signals for test period
            for idx in range(len(test_data)):
                current_data = pd.concat([train_data, test_data.iloc[:idx+1]])
                
                # Generate signal
                if signal_generator:
                    signal = signal_generator(symbol, current_data, test_data.index[idx])
                else:
                    signal = self._generate_ml_signal(symbol, current_data, ml_models)
                
                # Calculate actual movement
                if idx < len(test_data) - 1:
                    next_return = test_data.iloc[idx + 1]['Returns']
                    actual_direction = 'BUY' if next_return > 0.001 else 'SELL' if next_return < -0.001 else 'HOLD'
                    
                    results['predictions'].append(signal['action'])
                    results['actual_movements'].append(actual_direction)
                    
                    # Calculate profit/loss
                    if signal['action'] == 'BUY' and next_return > 0:
                        profit = next_return * signal['confidence']
                    elif signal['action'] == 'SELL' and next_return < 0:
                        profit = -next_return * signal['confidence']
                    else:
                        profit = -abs(next_return) * 0.5  # Penalty for wrong direction
                    
                    results['profits'].append(profit)
        
        # Calculate metrics
        if results['predictions']:
            results['accuracy'] = accuracy_score(
                results['actual_movements'],
                results['predictions']
            )
            results['total_profit'] = sum(results['profits'])
            results['sharpe_ratio'] = self._calculate_sharpe_ratio(results['profits'])
            results['max_drawdown'] = self._calculate_max_drawdown(results['profits'])
        
        return results
    
    def _generate_ml_signal(self, symbol: str, data: pd.DataFrame, ml_models: Optional[Dict]) -> Dict:
        """Generate signal using ML models"""
        
        # Prepare features
        features = self._prepare_ml_features(data)
        
        if not ml_models:
            # Use simple rule-based approach
            return self._generate_rule_based_signal(symbol, data)
        
        # Ensemble prediction from multiple models
        predictions = []
        confidences = []
        
        for model_name, model in ml_models.items():
            try:
                pred = model.predict(features.iloc[-1:])
                prob = model.predict_proba(features.iloc[-1:])[0]
                predictions.append(pred[0])
                confidences.append(max(prob))
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
        
        if predictions:
            # Majority vote
            action = max(set(predictions), key=predictions.count)
            confidence = np.mean(confidences)
        else:
            # Fallback to rule-based
            return self._generate_rule_based_signal(symbol, data)
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'timestamp': data.index[-1]
        }
    
    def _generate_rule_based_signal(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Generate signal using technical analysis rules"""
        
        latest = data.iloc[-1]
        
        # Score based on multiple indicators
        buy_score = 0
        sell_score = 0
        
        # RSI
        if latest['RSI'] < 30:
            buy_score += 2
        elif latest['RSI'] > 70:
            sell_score += 2
        
        # MACD
        if latest['MACD'] > latest['MACD_Signal']:
            buy_score += 1
        else:
            sell_score += 1
        
        # Price vs Moving Averages
        if latest['Close'] > latest['SMA_50']:
            buy_score += 1
        else:
            sell_score += 1
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            buy_score += 2
        elif latest['Close'] > latest['BB_Upper']:
            sell_score += 2
        
        # Volume
        if latest['Volume_Ratio'] > 1.5:
            if latest['Returns'] > 0:
                buy_score += 1
            else:
                sell_score += 1
        
        # Determine action
        if buy_score > sell_score + 1:
            action = 'BUY'
            confidence = buy_score / (buy_score + sell_score)
        elif sell_score > buy_score + 1:
            action = 'SELL'
            confidence = sell_score / (buy_score + sell_score)
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'timestamp': data.index[-1],
            'scores': {'buy': buy_score, 'sell': sell_score}
        }
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        
        feature_columns = [
            'Returns', 'Log_Returns', 'High_Low_Ratio', 'Close_Open_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Volume_Ratio', 'ATR', 'Stoch_K', 'Stoch_D'
        ]
        
        # Add price position features
        for period in [5, 10, 20, 50]:
            if f'SMA_{period}' in data.columns:
                data[f'Price_to_SMA_{period}'] = data['Close'] / data[f'SMA_{period}']
                feature_columns.append(f'Price_to_SMA_{period}')
        
        # Add momentum features
        for period in [1, 5, 10]:
            data[f'Momentum_{period}'] = data['Close'].pct_change(period)
            feature_columns.append(f'Momentum_{period}')
        
        return data[feature_columns].dropna()
    
    def _calculate_accuracy_metrics(self, symbol_results: Dict) -> Dict:
        """Calculate comprehensive accuracy metrics"""
        
        all_predictions = []
        all_actuals = []
        
        for symbol, results in symbol_results.items():
            if 'predictions' in results and 'actual_movements' in results:
                all_predictions.extend(results['predictions'])
                all_actuals.extend(results['actual_movements'])
        
        if not all_predictions:
            return {}
        
        # Calculate metrics
        accuracy = accuracy_score(all_actuals, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_actuals, all_predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_actuals, all_predictions)
        
        return {
            'overall_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'total_signals': len(all_predictions)
        }
    
    def _calculate_profit_metrics(self, symbol_results: Dict) -> Dict:
        """Calculate profit-related metrics"""
        
        total_profit = 0
        all_profits = []
        
        for symbol, results in symbol_results.items():
            if 'profits' in results:
                total_profit += results.get('total_profit', 0)
                all_profits.extend(results['profits'])
        
        if not all_profits:
            return {}
        
        return {
            'total_profit': total_profit,
            'average_profit_per_trade': np.mean(all_profits),
            'profit_factor': sum(p for p in all_profits if p > 0) / abs(sum(p for p in all_profits if p < 0)),
            'win_rate': len([p for p in all_profits if p > 0]) / len(all_profits),
            'average_win': np.mean([p for p in all_profits if p > 0]) if any(p > 0 for p in all_profits) else 0,
            'average_loss': np.mean([p for p in all_profits if p < 0]) if any(p < 0 for p in all_profits) else 0
        }
    
    def _calculate_risk_metrics(self, symbol_results: Dict) -> Dict:
        """Calculate risk-related metrics"""
        
        all_sharpe_ratios = []
        all_max_drawdowns = []
        
        for symbol, results in symbol_results.items():
            if 'sharpe_ratio' in results:
                all_sharpe_ratios.append(results['sharpe_ratio'])
            if 'max_drawdown' in results:
                all_max_drawdowns.append(results['max_drawdown'])
        
        return {
            'average_sharpe_ratio': np.mean(all_sharpe_ratios) if all_sharpe_ratios else 0,
            'average_max_drawdown': np.mean(all_max_drawdowns) if all_max_drawdowns else 0,
            'worst_drawdown': max(all_max_drawdowns) if all_max_drawdowns else 0
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not profits:
            return 0
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0
    
    def _generate_optimization_suggestions(self, results: Dict) -> List[str]:
        """Generate suggestions for improving accuracy"""
        
        suggestions = []
        
        # Check overall accuracy
        accuracy = results['accuracy_metrics'].get('overall_accuracy', 0)
        if accuracy < 0.6:
            suggestions.append("Consider adding more technical indicators or ML features")
            suggestions.append("Implement ensemble methods to combine multiple models")
            suggestions.append("Add market regime detection to adapt strategies")
        
        # Check profit metrics
        win_rate = results['profit_metrics'].get('win_rate', 0)
        if win_rate < 0.5:
            suggestions.append("Review and optimize entry/exit criteria")
            suggestions.append("Implement better risk management rules")
            suggestions.append("Consider adding stop-loss and take-profit levels")
        
        # Check risk metrics
        sharpe = results['risk_metrics'].get('average_sharpe_ratio', 0)
        if sharpe < 1.0:
            suggestions.append("Improve risk-adjusted returns by filtering low-confidence signals")
            suggestions.append("Implement position sizing based on signal confidence")
            suggestions.append("Add volatility-based filters")
        
        # Check for specific patterns
        if results['accuracy_metrics'].get('confusion_matrix'):
            cm = np.array(results['accuracy_metrics']['confusion_matrix'])
            if cm.shape[0] >= 3:  # BUY, SELL, HOLD
                # Check if model is biased towards HOLD
                hold_predictions = cm[:, 2].sum()
                total_predictions = cm.sum()
                if hold_predictions / total_predictions > 0.5:
                    suggestions.append("Model is too conservative - adjust confidence thresholds")
        
        return suggestions


class MLModelTrainer:
    """Train and optimize ML models for signal generation"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
    
    async def train_ensemble_models(
        self,
        training_data: pd.DataFrame,
        target_column: str = 'Signal'
    ) -> Dict:
        """Train ensemble of ML models"""
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        
        # Prepare features
        features = self._prepare_training_features(training_data)
        
        # Create target variable
        targets = self._create_target_variable(training_data)
        
        # Split data
        train_size = int(0.8 * len(features))
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = targets[:train_size], targets[train_size:]
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(features.columns, model.feature_importances_))
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score
            }
            
            self.models[name] = model
            
            logger.info(f"{name} - Train: {train_score:.4f}, Test: {test_score:.4f}")
        
        return results
    
    def _prepare_training_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        
        # Similar to backtest engine feature preparation
        feature_columns = []
        
        # Technical indicators
        technical_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Volume_Ratio', 'ATR', 'Stoch_K', 'Stoch_D'
        ]
        
        for col in technical_features:
            if col in data.columns:
                feature_columns.append(col)
        
        # Price-based features
        for period in [5, 10, 20, 50]:
            if f'SMA_{period}' in data.columns:
                data[f'Price_to_SMA_{period}'] = data['Close'] / data[f'SMA_{period}']
                feature_columns.append(f'Price_to_SMA_{period}')
        
        # Momentum features
        for period in [1, 5, 10]:
            data[f'Momentum_{period}'] = data['Close'].pct_change(period)
            feature_columns.append(f'Momentum_{period}')
        
        return data[feature_columns].dropna()
    
    def _create_target_variable(self, data: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Create target variable for classification"""
        
        # Calculate forward returns
        data['Forward_Return'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Create signals
        conditions = [
            data['Forward_Return'] > threshold,
            data['Forward_Return'] < -threshold
        ]
        choices = ['BUY', 'SELL']
        
        targets = pd.Series(
            np.select(conditions, choices, default='HOLD'),
            index=data.index
        )
        
        return targets[:-1]  # Remove last row (no forward return)
    
    def save_models(self, path: str = 'ml_models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}_model.pkl")
            logger.info(f"Saved {name} model")
    
    def load_models(self, path: str = 'ml_models/') -> Dict:
        """Load saved models"""
        import os
        
        models = {}
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith('_model.pkl'):
                    name = filename.replace('_model.pkl', '')
                    models[name] = joblib.load(f"{path}/{filename}")
                    logger.info(f"Loaded {name} model")
        
        self.models = models
        return models


async def run_comprehensive_backtest():
    """Run comprehensive backtest with ML validation"""
    
    # Initialize engines
    backtest_engine = AdvancedBacktestEngine()
    ml_trainer = MLModelTrainer()
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ']
    start_date = '2022-01-01'
    end_date = '2024-01-01'
    
    logger.info("Starting comprehensive backtest...")
    
    # First, train ML models on historical data
    logger.info("Training ML models...")
    
    # Fetch training data
    training_symbol = 'SPY'  # Use SPY for initial training
    ticker = yf.Ticker(training_symbol)
    training_data = ticker.history(start='2020-01-01', end='2022-01-01')
    
    # Add indicators
    engine = AdvancedBacktestEngine()
    training_data = engine._add_technical_indicators(training_data)
    
    # Train models
    ml_results = await ml_trainer.train_ensemble_models(training_data)
    
    # Save models
    ml_trainer.save_models()
    
    # Run backtest with trained models
    logger.info("Running backtest with ML models...")
    
    backtest_results = await backtest_engine.backtest_with_ml_validation(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        ml_models=ml_trainer.models
    )
    
    # Generate report
    report = {
        'backtest_period': f"{start_date} to {end_date}",
        'symbols_tested': symbols,
        'ml_models': list(ml_trainer.models.keys()),
        'accuracy_metrics': backtest_results['accuracy_metrics'],
        'profit_metrics': backtest_results['profit_metrics'],
        'risk_metrics': backtest_results['risk_metrics'],
        'optimization_suggestions': backtest_results['optimization_suggestions'],
        'feature_importance': ml_trainer.feature_importance
    }
    
    # Save report
    with open('backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE BACKTEST RESULTS")
    print("="*60)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"\nML Models Tested: {', '.join(ml_trainer.models.keys())}")
    
    print("\n📊 ACCURACY METRICS:")
    for metric, value in backtest_results['accuracy_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n💰 PROFIT METRICS:")
    for metric, value in backtest_results['profit_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n⚠️ RISK METRICS:")
    for metric, value in backtest_results['risk_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n🎯 OPTIMIZATION SUGGESTIONS:")
    for i, suggestion in enumerate(backtest_results['optimization_suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    print("\n📈 TOP FEATURE IMPORTANCE:")
    if ml_trainer.feature_importance:
        # Get average importance across models
        all_features = {}
        for model_features in ml_trainer.feature_importance.values():
            for feature, importance in model_features.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        avg_importance = {f: np.mean(imps) for f, imps in all_features.items()}
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
    
    print(f"\n📄 Detailed report saved to: backtest_report.json")
    
    return backtest_results


if __name__ == "__main__":
    # Run the comprehensive backtest
    asyncio.run(run_comprehensive_backtest()) 