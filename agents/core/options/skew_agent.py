"""
Skew Agent - Analyzes implied volatility skew patterns and their trading implications.
Tracks volatility skew, term structure, and skew-based arbitrage opportunities.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.interpolate import interp1d
from agents.common.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SkewAgent(BaseAgent):
    """Agent that analyzes implied volatility skew and generates skew-based signals."""
    
    def __init__(
        self,
        name: str = "Skew",
        skew_threshold: float = 0.15,
        extreme_skew_threshold: float = 0.25,
        term_structure_periods: List[int] = None,
        min_options_per_expiry: int = 5,
        skew_regression_window: int = 10
    ):
        """
        Initialize Skew agent.
        
        Args:
            name: Agent name
            skew_threshold: Threshold for significant skew
            extreme_skew_threshold: Threshold for extreme skew
            term_structure_periods: List of days for term structure analysis
            min_options_per_expiry: Minimum options needed per expiry
            skew_regression_window: Window for skew trend analysis
        """
        super().__init__(name=name, agent_type="volatility")
        self.skew_threshold = skew_threshold
        self.extreme_skew_threshold = extreme_skew_threshold
        self.term_structure_periods = term_structure_periods or [7, 14, 30, 60, 90]
        self.min_options_per_expiry = min_options_per_expiry
        self.skew_regression_window = skew_regression_window
        
    def calculate_volatility_skew(self, options_data: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
        """Calculate various measures of volatility skew."""
        try:
            skew_metrics = {}
            
            # Group options by expiry
            options_by_expiry = {}
            for option in options_data:
                expiry = option.get('expiry_date', 'unknown')
                if expiry not in options_by_expiry:
                    options_by_expiry[expiry] = []
                options_by_expiry[expiry].append(option)
            
            for expiry, expiry_options in options_by_expiry.items():
                if len(expiry_options) < self.min_options_per_expiry:
                    continue
                
                # Separate calls and puts
                calls = [opt for opt in expiry_options if opt.get('type', '').lower() == 'call']
                puts = [opt for opt in expiry_options if opt.get('type', '').lower() == 'put']
                
                if len(calls) < 3 or len(puts) < 3:
                    continue
                
                # Calculate skew metrics for this expiry
                expiry_skew = self.analyze_expiry_skew(calls, puts, spot_price)
                if expiry_skew:
                    skew_metrics[expiry] = expiry_skew
            
            return skew_metrics
            
        except Exception as e:
            logger.error(f"Volatility skew calculation failed: {str(e)}")
            return {}
    
    def analyze_expiry_skew(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Optional[Dict[str, Any]]:
        """Analyze skew for a specific expiry."""
        try:
            # Combine calls and puts, convert to standardized format
            all_options = []
            
            for call in calls:
                strike = call.get('strike', 0)
                iv = call.get('implied_volatility', 0)
                if strike > 0 and iv > 0:
                    moneyness = strike / spot_price
                    all_options.append({
                        'strike': strike,
                        'moneyness': moneyness,
                        'iv': iv,
                        'type': 'call',
                        'delta_equiv': moneyness - 1.0  # Approximate delta equivalent
                    })
            
            for put in puts:
                strike = put.get('strike', 0)
                iv = put.get('implied_volatility', 0)
                if strike > 0 and iv > 0:
                    moneyness = strike / spot_price
                    all_options.append({
                        'strike': strike,
                        'moneyness': moneyness,
                        'iv': iv,
                        'type': 'put',
                        'delta_equiv': moneyness - 1.0
                    })
            
            if len(all_options) < 5:
                return None
            
            # Sort by moneyness
            all_options.sort(key=lambda x: x['moneyness'])
            
            # Extract data for analysis
            moneyness_values = [opt['moneyness'] for opt in all_options]
            iv_values = [opt['iv'] for opt in all_options]
            
            # Calculate skew metrics
            skew_metrics = {}
            
            # 1. Put-Call Skew (traditional measure)
            otm_puts = [opt for opt in all_options if opt['type'] == 'put' and opt['moneyness'] < 0.95]
            otm_calls = [opt for opt in all_options if opt['type'] == 'call' and opt['moneyness'] > 1.05]
            
            if otm_puts and otm_calls:
                avg_put_iv = np.mean([opt['iv'] for opt in otm_puts])
                avg_call_iv = np.mean([opt['iv'] for opt in otm_calls])
                put_call_skew = avg_put_iv - avg_call_iv
                skew_metrics['put_call_skew'] = put_call_skew
            
            # 2. 25-Delta Risk Reversal (if we have enough data points)
            if len(all_options) >= 7:
                # Interpolate to get 25-delta equivalent strikes
                try:
                    f = interp1d(moneyness_values, iv_values, kind='linear', fill_value='extrapolate')
                    
                    # Approximate 25-delta strikes (rough approximation)
                    put_25d_moneyness = 0.90  # Approximate
                    call_25d_moneyness = 1.10  # Approximate
                    
                    if min(moneyness_values) <= put_25d_moneyness <= max(moneyness_values):
                        put_25d_iv = f(put_25d_moneyness)
                    else:
                        put_25d_iv = None
                        
                    if min(moneyness_values) <= call_25d_moneyness <= max(moneyness_values):
                        call_25d_iv = f(call_25d_moneyness)
                    else:
                        call_25d_iv = None
                    
                    if put_25d_iv is not None and call_25d_iv is not None:
                        risk_reversal_25d = put_25d_iv - call_25d_iv
                        skew_metrics['risk_reversal_25d'] = float(risk_reversal_25d)
                    
                except Exception:
                    pass
            
            # 3. Skew slope (regression of IV vs moneyness)
            if len(all_options) >= 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(moneyness_values, iv_values)
                skew_metrics['skew_slope'] = slope
                skew_metrics['skew_r_squared'] = r_value ** 2
            
            # 4. ATM volatility
            atm_options = [opt for opt in all_options if 0.98 <= opt['moneyness'] <= 1.02]
            if atm_options:
                atm_iv = np.mean([opt['iv'] for opt in atm_options])
                skew_metrics['atm_iv'] = atm_iv
            
            # 5. Volatility smile curvature
            if len(all_options) >= 7:
                # Fit quadratic to get curvature
                moneyness_array = np.array(moneyness_values)
                iv_array = np.array(iv_values)
                
                try:
                    # Fit: IV = a + b*moneyness + c*moneyness^2
                    X = np.vstack([np.ones(len(moneyness_array)), moneyness_array, moneyness_array**2]).T
                    coeffs = np.linalg.lstsq(X, iv_array, rcond=None)[0]
                    skew_metrics['smile_curvature'] = coeffs[2]  # Quadratic coefficient
                except Exception:
                    pass
            
            # 6. Wing spread (difference between extreme strikes)
            if len(all_options) >= 5:
                sorted_by_iv = sorted(all_options, key=lambda x: x['iv'])
                low_wing_iv = np.mean([opt['iv'] for opt in sorted_by_iv[:2]])
                high_wing_iv = np.mean([opt['iv'] for opt in sorted_by_iv[-2:]])
                wing_spread = high_wing_iv - low_wing_iv
                skew_metrics['wing_spread'] = wing_spread
            
            return skew_metrics
            
        except Exception as e:
            logger.error(f"Expiry skew analysis failed: {str(e)}")
            return None
    
    def analyze_term_structure(self, skew_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze skew term structure across different expiries."""
        try:
            if not skew_data:
                return {}
            
            term_structure = {}
            
            # Extract skew metrics by expiry
            expiry_skews = []
            for expiry, metrics in skew_data.items():
                # Try to parse expiry to get days to expiration
                # This would need actual date parsing in production
                try:
                    # Placeholder for days calculation
                    days_to_expiry = 30  # Would calculate from actual dates
                    
                    if 'put_call_skew' in metrics:
                        expiry_skews.append({
                            'expiry': expiry,
                            'days': days_to_expiry,
                            'skew': metrics['put_call_skew'],
                            'atm_iv': metrics.get('atm_iv', 0)
                        })
                except Exception:
                    continue
            
            if len(expiry_skews) < 2:
                return {}
            
            # Sort by days to expiry
            expiry_skews.sort(key=lambda x: x['days'])
            
            # Calculate term structure metrics
            term_structure['expiry_count'] = len(expiry_skews)
            
            # Short vs long term skew
            if len(expiry_skews) >= 2:
                short_term_skew = expiry_skews[0]['skew']
                long_term_skew = expiry_skews[-1]['skew']
                
                term_structure['short_term_skew'] = short_term_skew
                term_structure['long_term_skew'] = long_term_skew
                term_structure['term_skew_spread'] = long_term_skew - short_term_skew
            
            # Skew term structure slope
            if len(expiry_skews) >= 3:
                days_list = [exp['days'] for exp in expiry_skews]
                skew_list = [exp['skew'] for exp in expiry_skews]
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(days_list, skew_list)
                term_structure['term_slope'] = slope
                term_structure['term_r_squared'] = r_value ** 2
            
            # Volatility term structure
            atm_iv_list = [exp['atm_iv'] for exp in expiry_skews if exp['atm_iv'] > 0]
            if len(atm_iv_list) >= 2:
                term_structure['vol_term_structure'] = atm_iv_list
                term_structure['vol_contango'] = atm_iv_list[-1] > atm_iv_list[0]
            
            return term_structure
            
        except Exception as e:
            logger.error(f"Term structure analysis failed: {str(e)}")
            return {}
    
    def detect_skew_anomalies(self, skew_data: Dict[str, Any], term_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in volatility skew that may present trading opportunities."""
        try:
            anomalies = []
            
            # Check each expiry for anomalies
            for expiry, metrics in skew_data.items():
                anomaly_detected = False
                anomaly_type = ""
                severity = "normal"
                
                # Extreme skew detection
                if 'put_call_skew' in metrics:
                    skew_value = metrics['put_call_skew']
                    
                    if abs(skew_value) > self.extreme_skew_threshold:
                        anomaly_detected = True
                        anomaly_type = "extreme_skew"
                        severity = "high"
                        
                        if skew_value > 0:
                            direction = "negative_skew"  # Puts more expensive
                        else:
                            direction = "positive_skew"  # Calls more expensive
                    
                    elif abs(skew_value) > self.skew_threshold:
                        anomaly_detected = True
                        anomaly_type = "elevated_skew"
                        severity = "medium"
                        direction = "negative_skew" if skew_value > 0 else "positive_skew"
                
                # Smile curvature anomalies
                if 'smile_curvature' in metrics:
                    curvature = metrics['smile_curvature']
                    if abs(curvature) > 0.5:  # High curvature threshold
                        anomaly_detected = True
                        anomaly_type = "extreme_curvature"
                        severity = "medium"
                        direction = "high_curvature"
                
                # Wing spread anomalies
                if 'wing_spread' in metrics:
                    wing_spread = metrics['wing_spread']
                    if wing_spread > 0.3:  # High wing spread
                        anomaly_detected = True
                        anomaly_type = "wide_wings"
                        severity = "medium"
                        direction = "wide_spread"
                
                if anomaly_detected:
                    anomalies.append({
                        'expiry': expiry,
                        'anomaly_type': anomaly_type,
                        'severity': severity,
                        'direction': direction,
                        'metrics': metrics
                    })
            
            # Term structure anomalies
            if term_structure and 'term_skew_spread' in term_structure:
                term_spread = term_structure['term_skew_spread']
                
                if abs(term_spread) > 0.1:  # Significant term structure inversion/steepening
                    anomalies.append({
                        'expiry': 'term_structure',
                        'anomaly_type': 'term_structure_anomaly',
                        'severity': 'medium',
                        'direction': 'inverted' if term_spread < 0 else 'steep',
                        'metrics': term_structure
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Skew anomaly detection failed: {str(e)}")
            return []
    
    def generate_skew_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on skew analysis."""
        try:
            options_data = data.get('options_data', [])
            spot_price = data.get('spot_price', 0)
            
            if not options_data or not spot_price:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'no_data',
                    'reasoning': 'Insufficient options data'
                }
            
            # Calculate skew
            skew_data = self.calculate_volatility_skew(options_data, spot_price)
            
            # Analyze term structure
            term_structure = self.analyze_term_structure(skew_data)
            
            # Detect anomalies
            anomalies = self.detect_skew_anomalies(skew_data, term_structure)
            
            # Generate signals
            action = "hold"
            confidence = 0.0
            signal_type = "neutral"
            reasoning = []
            
            # High-severity anomalies generate stronger signals
            high_severity_anomalies = [a for a in anomalies if a['severity'] == 'high']
            medium_severity_anomalies = [a for a in anomalies if a['severity'] == 'medium']
            
            if high_severity_anomalies:
                primary_anomaly = high_severity_anomalies[0]
                anomaly_type = primary_anomaly['anomaly_type']
                direction = primary_anomaly['direction']
                
                if anomaly_type == "extreme_skew":
                    if direction == "negative_skew":
                        # Puts very expensive - sell put spreads or buy call spreads
                        action = "sell"
                        signal_type = "sell_put_skew"
                        reasoning.append("Extreme negative skew - puts overpriced")
                        confidence += 0.6
                        
                    elif direction == "positive_skew":
                        # Calls very expensive - sell call spreads or buy put spreads  
                        action = "buy"
                        signal_type = "sell_call_skew"
                        reasoning.append("Extreme positive skew - calls overpriced")
                        confidence += 0.6
                        
            elif medium_severity_anomalies:
                confidence += 0.3
                reasoning.append(f"Medium severity skew anomaly detected")
            
            # Term structure signals
            if term_structure and 'term_skew_spread' in term_structure:
                term_spread = term_structure['term_skew_spread']
                
                if abs(term_spread) > 0.08:
                    confidence += 0.2
                    if term_spread < 0:
                        reasoning.append("Inverted skew term structure")
                    else:
                        reasoning.append("Steep skew term structure")
            
            # Multiple anomalies boost confidence
            if len(anomalies) > 1:
                confidence *= 1.2
                reasoning.append(f"Multiple skew anomalies detected ({len(anomalies)})")
            
            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Skew signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process options data and generate skew-based signals."""
        try:
            if "options_data" not in data or "spot_price" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Options data or spot price not provided"}
                }
            
            # Generate signals
            signal_data = self.generate_skew_signals(data)
            
            # Get comprehensive analysis
            options_data = data["options_data"]
            spot_price = data["spot_price"]
            
            skew_data = self.calculate_volatility_skew(options_data, spot_price)
            term_structure = self.analyze_term_structure(skew_data)
            anomalies = self.detect_skew_anomalies(skew_data, term_structure)
            
            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "skew_data": skew_data,
                    "term_structure": term_structure,
                    "anomalies": anomalies
                }
            }
            
        except Exception as e:
            logger.error(f"Skew signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 