"""
Options Analysis: Comprehensive options market analysis and strategy generation.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class OptionsAnalysis:
    """Comprehensive options market analysis and strategy generation."""
    
    def __init__(self, data_fetcher):
        """Initialize OptionsAnalysis with data fetcher."""
        self.data_fetcher = data_fetcher
        logger.info("OptionsAnalysis initialized")
        
    async def analyze_options_chain(self, symbol: str) -> Dict:
        """Analyze options chain for trading opportunities."""
        try:
            # Fetch options chain data
            options_chain = await self.data_fetcher.fetch_options_chain(symbol)
            if options_chain.empty:
                return self._create_empty_analysis()
                
            # Calculate key metrics
            iv_analysis = self._analyze_implied_volatility(options_chain)
            volume_analysis = self._analyze_options_volume(options_chain)
            skew_analysis = self._analyze_volatility_skew(options_chain)
            flow_analysis = self._analyze_options_flow(options_chain)
            
            return {
                "iv_analysis": iv_analysis,
                "volume_analysis": volume_analysis,
                "skew_analysis": skew_analysis,
                "flow_analysis": flow_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Options chain analysis failed: {e}")
            return self._create_empty_analysis()
            
    def _analyze_implied_volatility(self, options_chain: pd.DataFrame) -> Dict:
        """Analyze implied volatility patterns."""
        try:
            # Calculate IV metrics
            atm_strike = options_chain["strike"].iloc[len(options_chain)//2]
            atm_iv = options_chain[options_chain["strike"] == atm_strike]["implied_volatility"].iloc[0]
            
            # Calculate IV percentiles
            iv_percentile = np.percentile(options_chain["implied_volatility"], 50)
            iv_rank = (atm_iv - options_chain["implied_volatility"].min()) / \
                     (options_chain["implied_volatility"].max() - options_chain["implied_volatility"].min())
            
            return {
                "atm_iv": atm_iv,
                "iv_percentile": iv_percentile,
                "iv_rank": iv_rank,
                "is_high": iv_percentile > 0.7,
                "is_low": iv_percentile < 0.3
            }
            
        except Exception as e:
            logger.error(f"IV analysis failed: {e}")
            return {}
            
    def _analyze_options_volume(self, options_chain: pd.DataFrame) -> Dict:
        """Analyze options volume patterns."""
        try:
            # Calculate volume metrics
            call_volume = options_chain[options_chain["type"] == "call"]["volume"].sum()
            put_volume = options_chain[options_chain["type"] == "put"]["volume"].sum()
            total_volume = call_volume + put_volume
            
            # Calculate volume ratios
            put_call_ratio = put_volume / call_volume if call_volume > 0 else float('inf')
            volume_oi_ratio = total_volume / options_chain["open_interest"].sum()
            
            return {
                "call_volume": call_volume,
                "put_volume": put_volume,
                "total_volume": total_volume,
                "put_call_ratio": put_call_ratio,
                "volume_oi_ratio": volume_oi_ratio,
                "is_high_volume": volume_oi_ratio > 1.5,
                "is_low_volume": volume_oi_ratio < 0.5
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {}
            
    def _analyze_volatility_skew(self, options_chain: pd.DataFrame) -> Dict:
        """Analyze volatility skew patterns."""
        try:
            # Calculate skew metrics
            atm_strike = options_chain["strike"].iloc[len(options_chain)//2]
            atm_iv = options_chain[options_chain["strike"] == atm_strike]["implied_volatility"].iloc[0]
            
            # Calculate put skew
            otm_puts = options_chain[
                (options_chain["type"] == "put") & 
                (options_chain["strike"] < atm_strike)
            ]
            put_skew = otm_puts["implied_volatility"].mean() - atm_iv
            
            # Calculate call skew
            otm_calls = options_chain[
                (options_chain["type"] == "call") & 
                (options_chain["strike"] > atm_strike)
            ]
            call_skew = otm_calls["implied_volatility"].mean() - atm_iv
            
            return {
                "put_skew": put_skew,
                "call_skew": call_skew,
                "is_put_skewed": put_skew > 0.1,
                "is_call_skewed": call_skew > 0.1,
                "skew_type": "put_skewed" if put_skew > 0.1 else "call_skewed" if call_skew > 0.1 else "neutral"
            }
            
        except Exception as e:
            logger.error(f"Skew analysis failed: {e}")
            return {}
            
    def _analyze_options_flow(self, options_chain: pd.DataFrame) -> Dict:
        """Analyze options flow patterns."""
        try:
            # Calculate flow metrics
            call_flow = options_chain[options_chain["type"] == "call"]["volume"].sum()
            put_flow = options_chain[options_chain["type"] == "put"]["volume"].sum()
            
            # Calculate flow ratios
            flow_ratio = call_flow / put_flow if put_flow > 0 else float('inf')
            
            # Calculate large trades
            large_trades = options_chain[options_chain["volume"] > options_chain["volume"].quantile(0.9)]
            large_call_trades = len(large_trades[large_trades["type"] == "call"])
            large_put_trades = len(large_trades[large_trades["type"] == "put"])
            
            return {
                "call_flow": call_flow,
                "put_flow": put_flow,
                "flow_ratio": flow_ratio,
                "large_call_trades": large_call_trades,
                "large_put_trades": large_put_trades,
                "is_bullish_flow": flow_ratio > 1.5,
                "is_bearish_flow": flow_ratio < 0.67
            }
            
        except Exception as e:
            logger.error(f"Flow analysis failed: {e}")
            return {}
            
    def _create_empty_analysis(self) -> Dict:
        """Create empty analysis result."""
        return {
            "iv_analysis": {},
            "volume_analysis": {},
            "skew_analysis": {},
            "flow_analysis": {},
            "timestamp": datetime.now().isoformat()
        }
        
    def generate_options_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate options trading strategies based on analysis."""
        strategies = []
        
        try:
            # Volatility-based strategies
            if analysis["iv_analysis"].get("is_high", False):
                strategies.extend(self._generate_high_vol_strategies(analysis))
            elif analysis["iv_analysis"].get("is_low", False):
                strategies.extend(self._generate_low_vol_strategies(analysis))
                
            # Flow-based strategies
            if analysis["flow_analysis"].get("is_bullish_flow", False):
                strategies.extend(self._generate_bullish_strategies(analysis))
            elif analysis["flow_analysis"].get("is_bearish_flow", False):
                strategies.extend(self._generate_bearish_strategies(analysis))
                
            # Skew-based strategies
            if analysis["skew_analysis"].get("is_put_skewed", False):
                strategies.extend(self._generate_put_skew_strategies(analysis))
            elif analysis["skew_analysis"].get("is_call_skewed", False):
                strategies.extend(self._generate_call_skew_strategies(analysis))
                
            return strategies
            
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return []
            
    def _generate_high_vol_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate strategies for high volatility environment."""
        return [
            {
                "type": "IRON_CONDOR",
                "confidence": 0.85,
                "setup_rules": {
                    "entry": "Sell OTM put spread and OTM call spread",
                    "exit": "Close at 50% max profit or if price approaches short strikes",
                    "adjustment": "Roll untested side if price moves towards it"
                }
            },
            {
                "type": "BUTTERFLY",
                "confidence": 0.80,
                "setup_rules": {
                    "entry": "Buy butterfly at expected price target",
                    "exit": "Close at 50% max profit or if price moves away from center",
                    "adjustment": "Roll to new center if price moves significantly"
                }
            }
        ]
        
    def _generate_low_vol_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate strategies for low volatility environment."""
        return [
            {
                "type": "CALENDAR",
                "confidence": 0.75,
                "setup_rules": {
                    "entry": "Sell near-term option, buy longer-term option",
                    "exit": "Close when near-term option expires or if volatility increases",
                    "adjustment": "Roll to new strikes if price moves significantly"
                }
            },
            {
                "type": "DIAGONAL",
                "confidence": 0.70,
                "setup_rules": {
                    "entry": "Sell near-term option, buy longer-term option at different strike",
                    "exit": "Close when near-term option expires or if trend reverses",
                    "adjustment": "Roll to new strikes if price moves significantly"
                }
            }
        ]
        
    def _generate_bullish_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate bullish strategies based on options flow."""
        return [
            {
                "type": "CALL",
                "confidence": 0.85,
                "setup_rules": {
                    "entry": "Buy calls when price pulls back to 20 EMA",
                    "exit": "Take profit at target or if trend weakens",
                    "adjustment": "Roll up if price moves strongly in favor"
                }
            },
            {
                "type": "BULL_CALL_SPREAD",
                "confidence": 0.80,
                "setup_rules": {
                    "entry": "Buy lower strike call, sell higher strike call",
                    "exit": "Close at 50% max profit or if price approaches short strike",
                    "adjustment": "Roll up if price moves strongly in favor"
                }
            }
        ]
        
    def _generate_bearish_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate bearish strategies based on options flow."""
        return [
            {
                "type": "PUT",
                "confidence": 0.85,
                "setup_rules": {
                    "entry": "Buy puts when price rallies to resistance",
                    "exit": "Take profit at target or if trend weakens",
                    "adjustment": "Roll down if price moves strongly in favor"
                }
            },
            {
                "type": "BEAR_PUT_SPREAD",
                "confidence": 0.80,
                "setup_rules": {
                    "entry": "Buy higher strike put, sell lower strike put",
                    "exit": "Close at 50% max profit or if price approaches short strike",
                    "adjustment": "Roll down if price moves strongly in favor"
                }
            }
        ]
        
    def _generate_put_skew_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate strategies for put-skewed environment."""
        return [
            {
                "type": "PUT_RATIO_SPREAD",
                "confidence": 0.85,
                "setup_rules": {
                    "entry": "Buy 1 ATM put, sell 2 OTM puts",
                    "exit": "Close at 50% max profit or if price moves significantly",
                    "adjustment": "Roll to new strikes if price moves significantly"
                }
            }
        ]
        
    def _generate_call_skew_strategies(self, analysis: Dict) -> List[Dict]:
        """Generate strategies for call-skewed environment."""
        return [
            {
                "type": "CALL_RATIO_SPREAD",
                "confidence": 0.85,
                "setup_rules": {
                    "entry": "Buy 1 ATM call, sell 2 OTM calls",
                    "exit": "Close at 50% max profit or if price moves significantly",
                    "adjustment": "Roll to new strikes if price moves significantly"
                }
            }
        ] 