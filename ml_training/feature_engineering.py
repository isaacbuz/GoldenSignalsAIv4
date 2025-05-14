import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Any, Tuple

class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering for trading strategies.
    """
    
    @staticmethod
    def extract_market_features(market_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract advanced market features for ML models.
        
        Args:
            market_data (Dict[str, np.ndarray]): Raw market data
        
        Returns:
            np.ndarray: Engineered features
        """
        # Price-based features
        prices = market_data['close']
        returns = np.diff(prices) / prices[:-1]
        
        features = [
            # Statistical moments
            np.mean(returns),
            np.std(returns),
            stats.skew(returns),
            stats.kurtosis(returns),
            
            # Rolling window statistics
            np.mean(prices[-20:]),  # 20-day moving average
            np.std(prices[-20:]),   # 20-day price volatility
            
            # Trend indicators
            np.polyfit(np.arange(len(prices[-20:])), prices[-20:], 1)[0],  # Linear trend
            
            # Volatility measures
            np.max(prices) - np.min(prices),  # Price range
            np.percentile(returns, 95) - np.percentile(returns, 5),  # Interquartile range
        ]
        
        return np.array(features)
    
    @staticmethod
    def extract_options_features(options_chain: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract advanced options-specific features.
        
        Args:
            options_chain (Dict[str, np.ndarray]): Options market data
        
        Returns:
            np.ndarray: Options-specific engineered features
        """
        # Implied volatility analysis
        call_vol = options_chain['call_implied_volatility']
        put_vol = options_chain['put_implied_volatility']
        
        features = [
            # Volatility metrics
            np.mean(call_vol),
            np.mean(put_vol),
            np.std(call_vol - put_vol),  # Volatility spread
            
            # Open interest analysis
            np.mean(options_chain['call_open_interest']),
            np.mean(options_chain['put_open_interest']),
            np.std(options_chain['call_open_interest'] - 
                   options_chain['put_open_interest']),
            
            # Strike price distribution
            np.mean(options_chain['strikes']),
            np.std(options_chain['strikes']),
            
            # Volatility skew
            stats.skew(call_vol),
            stats.skew(put_vol)
        ]
        
        return np.array(features)
    
    @classmethod
    def combine_features(
        cls, 
        market_data: Dict[str, np.ndarray], 
        options_chain: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine market and options features.
        
        Args:
            market_data (Dict[str, np.ndarray]): Market price data
            options_chain (Dict[str, np.ndarray]): Options market data
        
        Returns:
            np.ndarray: Combined engineered features
        """
        market_features = cls.extract_market_features(market_data)
        options_features = cls.extract_options_features(options_chain)
        
        return np.concatenate([market_features, options_features])

def main():
    """
    Demonstrate advanced feature engineering.
    """
    # Simulate market data
    np.random.seed(42)
    
    market_data = {
        'close': np.cumsum(np.random.normal(0.001, 0.05, 252)),
        'high': np.random.uniform(100, 110, 252),
        'low': np.random.uniform(90, 100, 252)
    }
    
    options_chain = {
        'strikes': np.linspace(90, 110, 20),
        'call_implied_volatility': np.random.uniform(0.1, 0.5, 20),
        'put_implied_volatility': np.random.uniform(0.1, 0.5, 20),
        'call_open_interest': np.random.randint(100, 10000, 20),
        'put_open_interest': np.random.randint(100, 10000, 20)
    }
    
    # Extract features
    market_features = AdvancedFeatureEngineer.extract_market_features(market_data)
    options_features = AdvancedFeatureEngineer.extract_options_features(options_chain)
    combined_features = AdvancedFeatureEngineer.combine_features(market_data, options_chain)
    
    print("Market Features:", market_features)
    print("\nOptions Features:", options_features)
    print("\nCombined Features:", combined_features)

if __name__ == '__main__':
    main()
