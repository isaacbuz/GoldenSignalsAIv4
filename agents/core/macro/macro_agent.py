"""
Macro Agent - Analyzes macroeconomic indicators and their impact on financial markets.
Handles interest rates, GDP, inflation, employment data, and economic surprises.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from ...common.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MacroAgent(BaseAgent):
    """Agent that analyzes macroeconomic data and generates macro-driven trading signals."""
    
    def __init__(
        self,
        name: str = "Macro",
        interest_rate_sensitivity: float = 2.0,
        inflation_threshold: float = 0.5,
        gdp_growth_threshold: float = 1.0,
        unemployment_threshold: float = 0.3,
        surprise_impact_multiplier: float = 1.5,
        lookback_period: int = 12  # months
    ):
        """
        Initialize Macro agent.
        
        Args:
            name: Agent name
            interest_rate_sensitivity: Sensitivity to rate changes
            inflation_threshold: Significant inflation change threshold
            gdp_growth_threshold: GDP growth change threshold
            unemployment_threshold: Unemployment change threshold
            surprise_impact_multiplier: Multiplier for economic surprises
            lookback_period: Period for trend analysis
        """
        super().__init__(name=name, agent_type="macro")
        self.interest_rate_sensitivity = interest_rate_sensitivity
        self.inflation_threshold = inflation_threshold
        self.gdp_growth_threshold = gdp_growth_threshold
        self.unemployment_threshold = unemployment_threshold
        self.surprise_impact_multiplier = surprise_impact_multiplier
        self.lookback_period = lookback_period
        
        # Economic indicator weights for different asset classes
        self.indicator_weights = {
            'equities': {
                'interest_rates': -0.8,  # Negative correlation
                'inflation': -0.6,
                'gdp_growth': 0.7,
                'employment': 0.5,
                'consumer_confidence': 0.4,
                'manufacturing_pmi': 0.6
            },
            'bonds': {
                'interest_rates': -0.9,  # Strong negative correlation
                'inflation': -0.8,
                'gdp_growth': -0.3,
                'employment': -0.4,
                'consumer_confidence': -0.2,
                'manufacturing_pmi': -0.3
            },
            'commodities': {
                'interest_rates': -0.5,
                'inflation': 0.8,  # Positive correlation
                'gdp_growth': 0.6,
                'employment': 0.3,
                'consumer_confidence': 0.4,
                'manufacturing_pmi': 0.7
            },
            'currency': {
                'interest_rates': 0.9,  # Strong positive correlation
                'inflation': 0.6,
                'gdp_growth': 0.8,
                'employment': 0.5,
                'consumer_confidence': 0.3,
                'manufacturing_pmi': 0.4
            }
        }
        
        # Expected ranges for economic indicators (for normalization)
        self.indicator_ranges = {
            'interest_rates': (-2.0, 2.0),  # Percentage change
            'inflation': (-1.0, 3.0),       # Annual percentage
            'gdp_growth': (-5.0, 8.0),      # Annual percentage
            'unemployment': (-2.0, 2.0),    # Percentage point change
            'consumer_confidence': (-20, 20), # Index change
            'manufacturing_pmi': (30, 70)   # PMI level
        }
    
    def calculate_macro_sentiment(self, macro_data: Dict[str, Any], asset_class: str = 'equities') -> Dict[str, Any]:
        """Calculate overall macro sentiment for a specific asset class."""
        try:
            if asset_class not in self.indicator_weights:
                asset_class = 'equities'  # Default
            
            weights = self.indicator_weights[asset_class]
            
            # Simple sentiment calculation based on available data
            total_sentiment = 0.0
            sentiment_components = {}
            
            # Interest rate impact
            if 'interest_rates' in macro_data:
                rate_change = macro_data['interest_rates'].get('recent_change', 0.0)
                rate_impact = max(-1.0, min(1.0, rate_change / 2.0))  # Normalize
                sentiment_components['interest_rates'] = rate_impact * weights.get('interest_rates', 0)
                total_sentiment += sentiment_components['interest_rates']
            
            # GDP growth impact
            if 'gdp_growth' in macro_data:
                gdp_growth = macro_data['gdp_growth'].get('current', 2.0)
                gdp_impact = max(-1.0, min(1.0, (gdp_growth - 2.0) / 3.0))  # Normalize around 2%
                sentiment_components['gdp_growth'] = gdp_impact * weights.get('gdp_growth', 0)
                total_sentiment += sentiment_components['gdp_growth']
            
            # Inflation impact
            if 'inflation' in macro_data:
                inflation_rate = macro_data['inflation'].get('current', 2.0)
                inflation_impact = max(-1.0, min(1.0, (2.0 - inflation_rate) / 2.0))  # Target around 2%
                sentiment_components['inflation'] = inflation_impact * weights.get('inflation', 0)
                total_sentiment += sentiment_components['inflation']
            
            # Normalize total sentiment
            total_sentiment = max(-1.0, min(1.0, total_sentiment))
            
            # Determine overall macro environment
            if total_sentiment > 0.3:
                macro_environment = 'positive'
            elif total_sentiment < -0.3:
                macro_environment = 'negative'
            else:
                macro_environment = 'neutral'
            
            return {
                'total_sentiment': total_sentiment,
                'macro_environment': macro_environment,
                'sentiment_components': sentiment_components
            }
            
        except Exception as e:
            logger.error(f"Macro sentiment calculation failed: {str(e)}")
            return {'total_sentiment': 0.0, 'macro_environment': 'neutral'}
    
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process and potentially modify a trading signal based on macro conditions."""
        try:
            # Get macro data from signal
            macro_data = signal.get('macro_data', {})
            asset_class = signal.get('asset_class', 'equities')
            
            if not macro_data:
                return signal  # Return unchanged if no macro data
            
            # Calculate macro sentiment
            sentiment_data = self.calculate_macro_sentiment(macro_data, asset_class)
            total_sentiment = sentiment_data['total_sentiment']
            
            # Modify signal based on macro sentiment
            original_action = signal.get('action', 'hold')
            original_confidence = signal.get('confidence', 0.0)
            
            # Macro adjustment factor
            macro_adjustment = 1.0 + (total_sentiment * 0.3)  # Up to 30% adjustment
            
            # Apply adjustment
            if abs(total_sentiment) > 0.2:  # Only adjust for significant macro signals
                adjusted_confidence = original_confidence * macro_adjustment
                adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
                
                signal['confidence'] = adjusted_confidence
                signal['macro_adjusted'] = True
                signal['macro_sentiment'] = total_sentiment
                signal['macro_environment'] = sentiment_data['macro_environment']
            
            return signal
            
        except Exception as e:
            logger.error(f"Macro signal processing failed: {str(e)}")
            return signal
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process macroeconomic data and generate macro-driven signals."""
        try:
            if "macro_data" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "No macro data provided"}
                }
            
            macro_data = data["macro_data"]
            asset_class = data.get("asset_class", "equities")
            
            # Calculate macro sentiment
            sentiment_data = self.calculate_macro_sentiment(macro_data, asset_class)
            total_sentiment = sentiment_data['total_sentiment']
            
            # Generate signal based on sentiment
            action = "hold"
            confidence = 0.0
            reasoning = []
            
            if abs(total_sentiment) > 0.4:
                if total_sentiment > 0:
                    action = "buy"
                    reasoning.append(f"Positive macro sentiment: {total_sentiment:.2f}")
                else:
                    action = "sell"
                    reasoning.append(f"Negative macro sentiment: {total_sentiment:.2f}")
                
                confidence = min(abs(total_sentiment), 1.0)
            
            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "macro_sentiment": sentiment_data,
                    "asset_class": asset_class,
                    "reasoning": reasoning
                }
            }
            
        except Exception as e:
            logger.error(f"Macro signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 