#!/usr/bin/env python3
"""
ML-Enhanced Backtesting System for GoldenSignalsAI
Incorporates best practices from QuantConnect, Backtrader, and industry standards
Enhanced with Phase 2 Signal Generation and Monitoring Integration
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

# Import our Phase 2 services
try:
    from src.services.signal_generation_engine import SignalGenerationEngine, TradingSignal
    from src.services.signal_filtering_pipeline import SignalFilteringPipeline, ConfidenceFilter, QualityScoreFilter
    from src.services.signal_monitoring_service import SignalMonitoringService, SignalOutcome
    from src.services.data_quality_validator import DataQualityValidator
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Phase 2 services not available. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLBacktestEngine:
    """
    Professional-grade ML backtesting engine inspired by QuantConnect and Backtrader
    Enhanced with Phase 2 signal generation and monitoring integration
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.backtest_results = {}

        # Initialize Phase 2 services if available
        if PHASE2_AVAILABLE:
            self.signal_engine = SignalGenerationEngine()
            self.filter_pipeline = SignalFilteringPipeline()
            self.monitor_service = SignalMonitoringService()
            self.quality_validator = DataQualityValidator()
            logger.info("✅ Phase 2 services integrated")
        else:
            self.signal_engine = None
            self.filter_pipeline = None
            self.monitor_service = None
            self.quality_validator = None

        # Track signal quality metrics
        self.signal_quality_metrics = {
            'confidence_scores': [],
            'quality_scores': [],
            'filtered_ratio': 0,
            'signal_accuracy': []
        }

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data with corporate actions adjustment"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            # Add additional data if available
            df['Symbol'] = symbol

            # Handle splits and dividends
            actions = ticker.actions
            if not actions.empty:
                df = df.join(actions, how='left')
                df['Dividends'] = df['Dividends'].fillna(0)
                df['Stock Splits'] = df['Stock Splits'].fillna(0)

            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering inspired by quantitative finance best practices
        """
        if df.empty:
            return df

        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()

        # Technical indicators using ta library
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['bb_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

        # Volume features
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['dollar_volume'] = df['Close'] * df['Volume']

        # Price position features
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']

        # Market microstructure
        df['spread'] = df['High'] - df['Low']
        df['spread_pct'] = df['spread'] / df['Close']

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['returns'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['returns'].rolling(window=window).max()

        # Target variable (next day return direction)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)

        # Drop NaN values
        df.dropna(inplace=True)

        return df

    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML models"""
        feature_cols = [col for col in df.columns if col not in [
            'target', 'Symbol', 'Dividends', 'Stock Splits', 'Date'
        ]]

        X = df[feature_cols].values
        y = df['target'].values

        return X, y

    def train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train multiple ML models for ensemble predictions"""
        models = {}

        # Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models['random_forest'] = rf

        # XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model

        # LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model

        # Neural Network
        logger.info("Training Neural Network...")
        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        nn.fit(X_train, y_train)
        models['neural_network'] = nn

        return models

    def walk_forward_validation(self, df: pd.DataFrame, n_splits: int = 5) -> Dict:
        """
        Walk-forward validation to avoid lookahead bias
        Similar to QuantConnect's approach
        """
        X, y = self.prepare_ml_data(df)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'returns': []
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train models
            models = self.train_ensemble_models(X_train_scaled, y_train)

            # Ensemble predictions
            predictions = []
            for name, model in models.items():
                pred = model.predict(X_test_scaled)
                predictions.append(pred)

            # Majority vote
            ensemble_pred = np.round(np.mean(predictions, axis=0))

            # Calculate metrics
            accuracy = accuracy_score(y_test, ensemble_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, ensemble_pred, average='binary'
            )

            results['accuracy'].append(accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)

            # Simulate returns
            test_df = df.iloc[test_idx].copy()
            test_df['prediction'] = ensemble_pred
            test_df['strategy_returns'] = test_df['prediction'] * test_df['returns']

            total_return = (1 + test_df['strategy_returns']).prod() - 1
            results['returns'].append(total_return)

        return results

    def calculate_backtest_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate professional backtesting metrics
        Inspired by QuantConnect and institutional standards
        """
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def simulate_trading_with_costs(self, df: pd.DataFrame, predictions: np.ndarray,
                                  commission: float = 0.001, slippage: float = 0.0005) -> pd.DataFrame:
        """
        Simulate trading with realistic costs
        """
        df = df.copy()
        df['prediction'] = predictions

        # Calculate positions
        df['position'] = df['prediction'].diff().fillna(df['prediction'])

        # Calculate costs
        df['commission_cost'] = abs(df['position']) * commission
        df['slippage_cost'] = abs(df['position']) * slippage

        # Calculate net returns
        df['gross_returns'] = df['prediction'] * df['returns']
        df['net_returns'] = df['gross_returns'] - df['commission_cost'] - df['slippage_cost']

        return df

    async def run_comprehensive_backtest(self, symbols: List[str],
                                       start_date: str = "2020-01-01",
                                       end_date: str = None) -> Dict:
        """
        Run comprehensive ML-enhanced backtest with Phase 2 integration
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_results = {}

        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Backtesting {symbol}")
            logger.info(f"{'='*50}")

            # Fetch and prepare data
            df = self.fetch_historical_data(symbol, start_date, end_date)
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                continue

            # Engineer features
            df = self.engineer_features(df)

            # Walk-forward validation
            wf_results = self.walk_forward_validation(df)

            # Calculate aggregate metrics
            avg_accuracy = np.mean(wf_results['accuracy'])
            avg_return = np.mean(wf_results['returns'])

            # Full backtest
            X, y = self.prepare_ml_data(df)

            # Train-test split (80-20)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale and train
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            models = self.train_ensemble_models(X_train_scaled, y_train)

            # Get predictions
            predictions = []
            for name, model in models.items():
                pred = model.predict(X_test_scaled)
                predictions.append(pred)

            ensemble_pred = np.round(np.mean(predictions, axis=0))

            # Simulate trading
            test_df = df.iloc[split_idx:].copy()
            trading_df = self.simulate_trading_with_costs(test_df, ensemble_pred)

            # Calculate metrics
            metrics = self.calculate_backtest_metrics(trading_df['net_returns'])

            # Phase 2 Integration: Generate and evaluate signals
            signal_quality_results = {}
            if self.signal_engine and len(test_df) > 0:
                signal_quality_results = await self._evaluate_signal_quality(
                    symbol, test_df, ensemble_pred
                )
                metrics.update(signal_quality_results)

            # Feature importance (from Random Forest)
            feature_names = [col for col in df.columns if col not in [
                'target', 'Symbol', 'Dividends', 'Stock Splits'
            ]]
            feature_importance = dict(zip(
                feature_names,
                models['random_forest'].feature_importances_
            ))

            # Store results
            all_results[symbol] = {
                'walk_forward_results': wf_results,
                'backtest_metrics': metrics,
                'feature_importance': sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],  # Top 10 features
                'avg_accuracy': avg_accuracy,
                'model_performance': {
                    'accuracy': accuracy_score(y_test, ensemble_pred),
                    'precision': precision_recall_fscore_support(
                        y_test, ensemble_pred, average='binary'
                    )[0],
                    'recall': precision_recall_fscore_support(
                        y_test, ensemble_pred, average='binary'
                    )[1]
                },
                'signal_quality': signal_quality_results
            }

            # Log results
            logger.info(f"\nResults for {symbol}:")
            logger.info(f"Average Walk-Forward Accuracy: {avg_accuracy:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Annual Return: {metrics['annual_return']:.2%}")

            if signal_quality_results:
                logger.info(f"Signal Quality Score: {signal_quality_results.get('avg_signal_quality', 0):.2f}")
                logger.info(f"Signal Filter Pass Rate: {signal_quality_results.get('filter_pass_rate', 0):.2%}")

        return all_results

    async def _evaluate_signal_quality(self, symbol: str, df: pd.DataFrame,
                                     predictions: np.ndarray) -> Dict:
        """
        Evaluate signal quality using Phase 2 services
        """
        results = {
            'signal_confidence_scores': [],
            'signal_quality_scores': [],
            'filtered_signals': 0,
            'total_signals': 0,
            'signal_performance': []
        }

        try:
            # Generate signals for each prediction
            signals = []
            for i, (idx, row) in enumerate(df.iterrows()):
                if i >= len(predictions):
                    break

                # Generate signal using our engine
                signal_data = await self.signal_engine.generate_signal(
                    symbol=symbol,
                    current_price=row['Close'],
                    volume=row['Volume'],
                    prediction=predictions[i]
                )

                if signal_data:
                    signals.append(signal_data)
                    results['signal_confidence_scores'].append(signal_data.confidence)

                    # Track quality if validator available
                    if self.quality_validator:
                        quality_report = await self.quality_validator.validate_data(
                            pd.DataFrame([row]), symbol
                        )
                        results['signal_quality_scores'].append(
                            quality_report.accuracy
                        )

            results['total_signals'] = len(signals)

            # Filter signals through pipeline
            if self.filter_pipeline and signals:
                filtered_signals = await self.filter_pipeline.filter_signals(signals)
                results['filtered_signals'] = len(filtered_signals)

                # Track filtered signal performance
                if self.monitor_service:
                    for signal in filtered_signals:
                        # Simulate entry
                        await self.monitor_service.track_signal_entry(
                            signal_id=signal.id,
                            actual_entry_price=signal.entry_price
                        )

                        # Simulate exit (using next day's price)
                        exit_idx = df.index.get_loc(signal.timestamp) + 1
                        if exit_idx < len(df):
                            exit_price = df.iloc[exit_idx]['Close']
                            outcome = await self.monitor_service.track_signal_exit(
                                signal_id=signal.id,
                                exit_price=exit_price,
                                exit_reason="backtest_simulation"
                            )

                            if outcome:
                                results['signal_performance'].append({
                                    'signal_id': signal.id,
                                    'profit_loss': outcome.profit_loss,
                                    'profit_loss_pct': outcome.profit_loss_pct,
                                    'holding_period': outcome.holding_period
                                })

            # Calculate aggregate metrics
            if results['signal_confidence_scores']:
                results['avg_signal_confidence'] = np.mean(results['signal_confidence_scores'])
            if results['signal_quality_scores']:
                results['avg_signal_quality'] = np.mean(results['signal_quality_scores'])
            if results['total_signals'] > 0:
                results['filter_pass_rate'] = results['filtered_signals'] / results['total_signals']

            # Calculate signal performance metrics
            if results['signal_performance']:
                pl_values = [p['profit_loss_pct'] for p in results['signal_performance']]
                results['signal_win_rate'] = sum(1 for pl in pl_values if pl > 0) / len(pl_values)
                results['signal_avg_return'] = np.mean(pl_values)
                results['signal_sharpe'] = np.mean(pl_values) / np.std(pl_values) * np.sqrt(252) if np.std(pl_values) > 0 else 0

        except Exception as e:
            logger.error(f"Error evaluating signal quality: {e}")

        return results

    def get_signal_quality_summary(self) -> Dict:
        """
        Get summary of signal quality metrics across all backtests
        """
        if not self.signal_quality_metrics['confidence_scores']:
            return {}

        return {
            'avg_confidence': np.mean(self.signal_quality_metrics['confidence_scores']),
            'avg_quality': np.mean(self.signal_quality_metrics['quality_scores']),
            'overall_filter_ratio': self.signal_quality_metrics['filtered_ratio'],
            'signal_accuracy_history': self.signal_quality_metrics['signal_accuracy'],
            'confidence_distribution': {
                'min': np.min(self.signal_quality_metrics['confidence_scores']),
                'max': np.max(self.signal_quality_metrics['confidence_scores']),
                'std': np.std(self.signal_quality_metrics['confidence_scores'])
            }
        }


class SignalAccuracyImprover:
    """
    Improves signal accuracy using ML insights from backtesting
    """

    def __init__(self):
        self.backtest_engine = MLBacktestEngine()
        self.best_models = {}
        self.feature_weights = {}

    async def improve_signals(self, symbols: List[str]) -> Dict:
        """
        Run backtests and extract insights to improve signals
        """
        # Run comprehensive backtest
        results = await self.backtest_engine.run_comprehensive_backtest(symbols)

        # Extract best practices
        improvements = {
            'recommended_features': self._extract_best_features(results),
            'optimal_parameters': self._extract_optimal_parameters(results),
            'risk_management': self._generate_risk_rules(results),
            'signal_filters': self._generate_signal_filters(results)
        }

        return improvements

    def _extract_best_features(self, results: Dict) -> List[str]:
        """Extract most important features across all symbols"""
        all_features = {}

        for symbol, data in results.items():
            for feature, importance in data['feature_importance']:
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)

        # Average importance
        avg_importance = {
            feature: np.mean(scores)
            for feature, scores in all_features.items()
        }

        # Return top features
        return sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]

    def _extract_optimal_parameters(self, results: Dict) -> Dict:
        """Extract optimal trading parameters"""
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []

        for symbol, data in results.items():
            metrics = data['backtest_metrics']
            sharpe_ratios.append(metrics['sharpe_ratio'])
            max_drawdowns.append(abs(metrics['max_drawdown']))
            win_rates.append(metrics['win_rate'])

        return {
            'min_sharpe_ratio': np.percentile(sharpe_ratios, 25),
            'max_acceptable_drawdown': np.percentile(max_drawdowns, 75),
            'min_win_rate': np.percentile(win_rates, 25),
            'recommended_position_size': 0.1 / np.mean(max_drawdowns)  # Kelly-inspired
        }

    def _generate_risk_rules(self, results: Dict) -> Dict:
        """Generate risk management rules"""
        volatilities = []
        profit_factors = []

        for symbol, data in results.items():
            metrics = data['backtest_metrics']
            volatilities.append(metrics['volatility'])
            profit_factors.append(metrics['profit_factor'])

        return {
            'max_volatility': np.percentile(volatilities, 75),
            'min_profit_factor': max(1.2, np.percentile(profit_factors, 25)),
            'stop_loss': np.mean([abs(data['backtest_metrics']['avg_loss']) for _, data in results.items()]) * 2,
            'take_profit': np.mean([data['backtest_metrics']['avg_win'] for _, data in results.items()]) * 1.5
        }

    def _generate_signal_filters(self, results: Dict) -> List[Dict]:
        """Generate signal filtering rules based on backtest results"""
        filters = []

        # Volume filter
        filters.append({
            'name': 'volume_filter',
            'condition': 'volume > volume_sma',
            'rationale': 'Higher volume indicates stronger conviction'
        })

        # Volatility filter
        filters.append({
            'name': 'volatility_filter',
            'condition': 'volatility < volatility.quantile(0.75)',
            'rationale': 'Avoid extremely volatile periods'
        })

        # Trend filter
        filters.append({
            'name': 'trend_filter',
            'condition': 'sma_20 > sma_50',
            'rationale': 'Trade with the trend'
        })

        # RSI filter
        filters.append({
            'name': 'rsi_filter',
            'condition': '30 < rsi < 70',
            'rationale': 'Avoid overbought/oversold conditions'
        })

        return filters


async def main():
    """
    Demonstrate ML-enhanced backtesting and signal improvement
    """
    improver = SignalAccuracyImprover()

    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']

    logger.info("Starting ML-enhanced backtesting and signal improvement...")
    logger.info("This incorporates best practices from QuantConnect, Backtrader, and industry standards")

    # Run improvement analysis
    improvements = await improver.improve_signals(symbols)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'ml_backtest_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(improvements, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_file}")

    # Display recommendations
    logger.info("\n" + "="*50)
    logger.info("SIGNAL IMPROVEMENT RECOMMENDATIONS")
    logger.info("="*50)

    logger.info("\nTop Features to Use:")
    for feature, importance in improvements['recommended_features'][:10]:
        logger.info(f"  - {feature}: {importance:.4f}")

    logger.info("\nOptimal Parameters:")
    for param, value in improvements['optimal_parameters'].items():
        logger.info(f"  - {param}: {value:.4f}")

    logger.info("\nRisk Management Rules:")
    for rule, value in improvements['risk_management'].items():
        logger.info(f"  - {rule}: {value:.4f}")

    logger.info("\nSignal Filters:")
    for filter_rule in improvements['signal_filters']:
        logger.info(f"  - {filter_rule['name']}: {filter_rule['condition']}")
        logger.info(f"    Rationale: {filter_rule['rationale']}")


if __name__ == "__main__":
    asyncio.run(main())
