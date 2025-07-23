"""
Application Services Facade
Provides a clean interface to all application services
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.container import get_container, initialize_container
from src.models.domain.signal import Signal
from src.services.signal_generation_engine import TradingSignal

logger = logging.getLogger(__name__)


class ApplicationServices:
    """
    Facade for all application services
    Provides a unified interface for the API layer
    """
    
    def __init__(self):
        self.container = initialize_container()
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize service references"""
        self.signal_engine = self.container.get("signal_engine")
        self.signal_filter = self.container.get("signal_filter")
        self.signal_monitor = self.container.get("signal_monitor")
        self.market_data = self.container.get("market_data")
        self.risk_manager = self.container.get("risk_manager")
        self.portfolio = self.container.get("portfolio")
        self.data_validator = self.container.get("data_validator")
        
    async def generate_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Generate trading signals for given symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            List of signal dictionaries
        """
        try:
            # Generate raw signals
            signals = await self.signal_engine.generate_signals(symbols)
            
            # Filter signals
            if signals:
                filtered_signals = await self.signal_filter.filter_signals(signals)
            else:
                filtered_signals = []
                
            # Convert to dict format
            return [signal.to_dict() for signal in filtered_signals]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
            
    async def get_market_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> Dict[str, Any]:
        """
        Get market data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, etc.)
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            Market data dictionary
        """
        try:
            data = await self.market_data.get_market_data(symbol, period, interval)
            return {
                "symbol": symbol,
                "data": data.to_dict('records') if data is not None else [],
                "period": period,
                "interval": interval,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {
                "symbol": symbol,
                "data": [],
                "error": str(e)
            }
            
    async def analyze_risk(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze risk for given signals
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Risk analysis results
        """
        try:
            # Convert dicts back to TradingSignal objects
            signal_objects = []
            for sig_dict in signals:
                signal = TradingSignal(
                    id=sig_dict['id'],
                    symbol=sig_dict['symbol'],
                    action=sig_dict['action'],
                    confidence=sig_dict['confidence'],
                    price=sig_dict['price'],
                    timestamp=datetime.fromisoformat(sig_dict['timestamp']),
                    reason=sig_dict['reason'],
                    indicators=sig_dict['indicators'],
                    risk_level=sig_dict['risk_level'],
                    entry_price=sig_dict['entry_price'],
                    stop_loss=sig_dict['stop_loss'],
                    take_profit=sig_dict['take_profit'],
                    metadata=sig_dict.get('metadata', {}),
                    quality_score=sig_dict.get('quality_score', 0.0)
                )
                signal_objects.append(signal)
                
            return await self.risk_manager.analyze_signals(signal_objects)
            
        except Exception as e:
            logger.error(f"Error analyzing risk: {e}")
            return {"error": str(e)}
            
    async def simulate_portfolio(self, signals: List[Dict[str, Any]], initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Simulate portfolio performance with given signals
        
        Args:
            signals: List of signal dictionaries
            initial_capital: Starting capital
            
        Returns:
            Portfolio simulation results
        """
        try:
            return await self.portfolio.simulate(signals, initial_capital)
        except Exception as e:
            logger.error(f"Error simulating portfolio: {e}")
            return {"error": str(e)}
            
    async def get_signal_performance(self, signal_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific signal
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Performance metrics
        """
        try:
            return await self.signal_monitor.get_signal_performance(signal_id)
        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            return {"error": str(e)}
            
    async def validate_data_quality(self, symbol: str) -> Dict[str, Any]:
        """
        Validate data quality for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Data quality report
        """
        try:
            data, source = await self.data_validator.get_market_data_with_fallback(symbol)
            if data is not None:
                report = self.data_validator.validate_market_data(data, symbol)
                return {
                    "symbol": symbol,
                    "is_valid": report.is_valid,
                    "issues": report.issues,
                    "score": report.overall_score,
                    "source": source
                }
            else:
                return {
                    "symbol": symbol,
                    "is_valid": False,
                    "issues": ["No data available"],
                    "score": 0.0
                }
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return {
                "symbol": symbol,
                "is_valid": False,
                "error": str(e)
            }


# Singleton instance
_services = None


def get_services() -> ApplicationServices:
    """Get the application services instance"""
    global _services
    if _services is None:
        _services = ApplicationServices()
    return _services 