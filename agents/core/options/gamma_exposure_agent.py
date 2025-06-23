"""
Gamma Exposure Agent - Analyzes gamma positioning and its impact on market dynamics.
Tracks dealer gamma exposure, gamma levels, and potential market pinning effects.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from scipy.optimize import brentq
import math
from ...common.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class GammaExposureAgent(BaseAgent):
    """Agent that analyzes gamma exposure and its impact on market behavior."""
    
    def __init__(
        self,
        name: str = "GammaExposure",
        gamma_threshold: float = 100000,  # Gamma exposure threshold
        pin_proximity_threshold: float = 0.02,  # 2% proximity to strike
        large_gamma_multiplier: float = 2.0,
        days_to_expiry_weight: float = 0.5,
        min_open_interest: int = 100
    ):
        """
        Initialize Gamma Exposure agent.
        
        Args:
            name: Agent name
            gamma_threshold: Threshold for significant gamma exposure
            pin_proximity_threshold: Price proximity to strike for pinning
            large_gamma_multiplier: Multiplier for large gamma positions
            days_to_expiry_weight: Weight for time decay in gamma calculation
            min_open_interest: Minimum open interest to consider
        """
        super().__init__()
        self.name = name
        self.agent_type = "options"
        self.gamma_threshold = gamma_threshold
        self.pin_proximity_threshold = pin_proximity_threshold
        self.large_gamma_multiplier = large_gamma_multiplier
        self.days_to_expiry_weight = days_to_expiry_weight
        self.min_open_interest = min_open_interest
        
    def calculate_black_scholes_gamma(
        self, 
        spot: float, 
        strike: float, 
        time_to_expiry: float, 
        volatility: float, 
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate Black-Scholes gamma for an option."""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return 0.0
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry)
            )
            
            gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
            
            return gamma
            
        except Exception as e:
            logger.error(f"Black-Scholes gamma calculation failed: {str(e)}")
            return 0.0
    
    def calculate_dealer_gamma_exposure(self, options_data: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
        """Calculate net dealer gamma exposure across all strikes."""
        try:
            total_call_gamma = 0.0
            total_put_gamma = 0.0
            gamma_by_strike = {}
            significant_levels = []
            
            for option in options_data:
                strike = option.get('strike', 0)
                option_type = option.get('type', '').lower()  # 'call' or 'put'
                open_interest = option.get('open_interest', 0)
                volume = option.get('volume', 0)
                time_to_expiry = option.get('time_to_expiry', 0)
                implied_vol = option.get('implied_volatility', 0.2)
                
                if open_interest < self.min_open_interest or time_to_expiry <= 0:
                    continue
                
                # Calculate gamma
                gamma = self.calculate_black_scholes_gamma(
                    spot_price, strike, time_to_expiry, implied_vol
                )
                
                # Dealer is typically short options (opposite of customer flow)
                # Assume customers are net long options, so dealers are net short
                dealer_position_multiplier = -1.0
                
                # Weight by open interest and time decay
                time_weight = 1.0 + self.days_to_expiry_weight * (1.0 / max(time_to_expiry, 0.01))
                position_gamma = gamma * open_interest * 100 * dealer_position_multiplier * time_weight
                
                if option_type == 'call':
                    total_call_gamma += position_gamma
                elif option_type == 'put':
                    total_put_gamma += position_gamma
                
                # Track by strike
                if strike not in gamma_by_strike:
                    gamma_by_strike[strike] = 0.0
                gamma_by_strike[strike] += position_gamma
                
                # Check for significant gamma levels
                if abs(position_gamma) > self.gamma_threshold:
                    significant_levels.append({
                        'strike': strike,
                        'gamma_exposure': position_gamma,
                        'type': option_type,
                        'proximity': abs(spot_price - strike) / spot_price,
                        'open_interest': open_interest
                    })
            
            # Net gamma exposure
            net_gamma = total_call_gamma + total_put_gamma
            
            # Find gamma flip point (where net gamma changes sign)
            gamma_flip_level = self.find_gamma_flip_point(gamma_by_strike, spot_price)
            
            return {
                'net_gamma_exposure': net_gamma,
                'call_gamma': total_call_gamma,
                'put_gamma': total_put_gamma,
                'gamma_by_strike': gamma_by_strike,
                'significant_levels': significant_levels,
                'gamma_flip_level': gamma_flip_level,
                'current_spot': spot_price
            }
            
        except Exception as e:
            logger.error(f"Dealer gamma exposure calculation failed: {str(e)}")
            return {}
    
    def find_gamma_flip_point(self, gamma_by_strike: Dict[float, float], spot_price: float) -> Optional[float]:
        """Find the price level where net gamma exposure flips sign."""
        try:
            if not gamma_by_strike:
                return None
            
            strikes = sorted(gamma_by_strike.keys())
            
            # Calculate cumulative gamma at each strike level
            cumulative_gamma = 0.0
            flip_candidates = []
            
            for strike in strikes:
                cumulative_gamma += gamma_by_strike[strike]
                
                # Look for sign changes
                if len(flip_candidates) == 0:
                    flip_candidates.append((strike, cumulative_gamma))
                else:
                    prev_gamma = flip_candidates[-1][1]
                    if (prev_gamma > 0 > cumulative_gamma) or (prev_gamma < 0 < cumulative_gamma):
                        flip_candidates.append((strike, cumulative_gamma))
            
            # Find the flip point closest to current spot
            if flip_candidates:
                closest_flip = min(flip_candidates, key=lambda x: abs(x[0] - spot_price))
                return float(closest_flip[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Gamma flip point calculation failed: {str(e)}")
            return None
    
    def analyze_gamma_pinning_risk(self, gamma_data: Dict[str, Any], spot_price: float) -> Dict[str, Any]:
        """Analyze risk of price pinning due to gamma exposure."""
        try:
            gamma_by_strike = gamma_data.get('gamma_by_strike', {})
            significant_levels = gamma_data.get('significant_levels', [])
            
            # Find strikes with high gamma exposure near current price
            pinning_candidates = []
            
            for level in significant_levels:
                strike = level['strike']
                proximity = level['proximity']
                gamma_exposure = level['gamma_exposure']
                
                if proximity <= self.pin_proximity_threshold:
                    pinning_strength = abs(gamma_exposure) / proximity if proximity > 0 else float('inf')
                    
                    pinning_candidates.append({
                        'strike': strike,
                        'gamma_exposure': gamma_exposure,
                        'proximity': proximity,
                        'pinning_strength': pinning_strength,
                        'direction': 'attractive' if gamma_exposure < 0 else 'repulsive'
                    })
            
            # Sort by pinning strength
            pinning_candidates.sort(key=lambda x: x['pinning_strength'], reverse=True)
            
            # Determine overall pinning effect
            if pinning_candidates:
                strongest_pin = pinning_candidates[0]
                pin_level = strongest_pin['strike']
                pin_strength = strongest_pin['pinning_strength']
                
                # Classify pinning strength
                if pin_strength > 1000000:  # Very high threshold
                    pin_classification = 'very_strong'
                elif pin_strength > 500000:
                    pin_classification = 'strong'
                elif pin_strength > 100000:
                    pin_classification = 'moderate'
                else:
                    pin_classification = 'weak'
                    
                return {
                    'pinning_detected': True,
                    'pin_level': pin_level,
                    'pin_strength': pin_strength,
                    'pin_classification': pin_classification,
                    'pinning_candidates': pinning_candidates[:3],  # Top 3
                    'distance_to_pin': abs(spot_price - pin_level) / spot_price
                }
            
            return {
                'pinning_detected': False,
                'pin_level': None,
                'pin_strength': 0.0,
                'pin_classification': 'none'
            }
            
        except Exception as e:
            logger.error(f"Gamma pinning analysis failed: {str(e)}")
            return {'pinning_detected': False}
    
    def calculate_gamma_impact_on_volatility(self, gamma_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how gamma exposure affects realized volatility."""
        try:
            net_gamma = gamma_data.get('net_gamma_exposure', 0)
            gamma_flip_level = gamma_data.get('gamma_flip_level')
            current_spot = gamma_data.get('current_spot', 0)
            
            # Gamma impact on volatility depends on position relative to flip point
            if gamma_flip_level and current_spot:
                distance_from_flip = (current_spot - gamma_flip_level) / current_spot
                
                # When dealers are short gamma (negative), they amplify moves
                # When dealers are long gamma (positive), they dampen moves
                if net_gamma < 0:
                    vol_impact = 'amplifying'  # Destabilizing
                    impact_magnitude = min(abs(net_gamma) / 1000000, 2.0)  # Cap at 2x
                elif net_gamma > 0:
                    vol_impact = 'dampening'  # Stabilizing
                    impact_magnitude = min(abs(net_gamma) / 1000000, 0.5)  # Max 50% reduction
                else:
                    vol_impact = 'neutral'
                    impact_magnitude = 0.0
                
                return {
                    'volatility_impact': vol_impact,
                    'impact_magnitude': impact_magnitude,
                    'distance_from_flip': distance_from_flip,
                    'regime': 'short_gamma' if net_gamma < 0 else 'long_gamma' if net_gamma > 0 else 'neutral'
                }
            
            return {
                'volatility_impact': 'neutral',
                'impact_magnitude': 0.0,
                'regime': 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Gamma volatility impact calculation failed: {str(e)}")
            return {'volatility_impact': 'neutral', 'impact_magnitude': 0.0}
    
    def generate_gamma_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on gamma analysis."""
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
            
            # Calculate gamma exposure
            gamma_data = self.calculate_dealer_gamma_exposure(options_data, spot_price)
            
            # Analyze pinning risk
            pinning_analysis = self.analyze_gamma_pinning_risk(gamma_data, spot_price)
            
            # Calculate volatility impact
            vol_impact = self.calculate_gamma_impact_on_volatility(gamma_data)
            
            # Generate signals
            action = "hold"
            confidence = 0.0
            signal_type = "neutral"
            reasoning = []
            
            net_gamma = gamma_data.get('net_gamma_exposure', 0)
            
            # Signal based on gamma regime
            if abs(net_gamma) > self.gamma_threshold:
                if net_gamma < 0:
                    # Dealers short gamma - expect higher volatility
                    signal_type = "short_gamma_regime"
                    reasoning.append("Dealers short gamma - expect volatility expansion")
                    confidence += 0.4
                    
                    # In short gamma regime, consider volatility trades
                    action = "buy"  # Buy volatility/options
                    
                elif net_gamma > 0:
                    # Dealers long gamma - expect lower volatility  
                    signal_type = "long_gamma_regime"
                    reasoning.append("Dealers long gamma - expect volatility compression")
                    confidence += 0.4
                    
                    # In long gamma regime, consider selling volatility
                    action = "sell"  # Sell volatility/options
            
            # Pinning effects
            if pinning_analysis.get('pinning_detected'):
                pin_level = pinning_analysis['pin_level']
                pin_classification = pinning_analysis['pin_classification']
                
                if pin_classification in ['strong', 'very_strong']:
                    confidence += 0.3
                    reasoning.append(f"Strong gamma pinning expected near {pin_level}")
                    
                    # Pinning suggests range-bound behavior
                    if action == "hold":
                        action = "sell"  # Sell options due to pinning
                        signal_type = "gamma_pinning"
            
            # Gamma flip level proximity
            gamma_flip = gamma_data.get('gamma_flip_level')
            if gamma_flip:
                distance_to_flip = abs(spot_price - gamma_flip) / spot_price
                if distance_to_flip < 0.05:  # Within 5%
                    confidence += 0.2
                    reasoning.append(f"Near gamma flip level at {gamma_flip}")
            
            # Volatility impact consideration
            if vol_impact['volatility_impact'] == 'amplifying':
                reasoning.append("Gamma profile amplifies price moves")
            elif vol_impact['volatility_impact'] == 'dampening':
                reasoning.append("Gamma profile dampens price moves")
            
            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Gamma signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process options data and generate gamma exposure signals."""
        try:
            if "options_data" not in data or "spot_price" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Options data or spot price not provided"}
                }
            
            # Generate signals
            signal_data = self.generate_gamma_signals(data)
            
            # Get comprehensive analysis
            options_data = data["options_data"]
            spot_price = data["spot_price"]
            
            gamma_data = self.calculate_dealer_gamma_exposure(options_data, spot_price)
            pinning_analysis = self.analyze_gamma_pinning_risk(gamma_data, spot_price)
            vol_impact = self.calculate_gamma_impact_on_volatility(gamma_data)
            
            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "gamma_exposure": gamma_data,
                    "pinning_analysis": pinning_analysis,
                    "volatility_impact": vol_impact
                }
            }
            
        except Exception as e:
            logger.error(f"Gamma exposure signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and potentially modify a trading signal.
        
        Args:
            signal (Dict[str, Any]): Trading signal to process.
        
        Returns:
            Dict[str, Any]: Processed trading signal with potential modifications.
        """
        # Default implementation: return signal as-is
        logger.info(f"Processing signal: {signal}")
        return signal 