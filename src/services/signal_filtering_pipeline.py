"""
Signal Filtering Pipeline for GoldenSignalsAI V2
Implements multi-stage filtering to ensure only high-quality signals
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.services.signal_generation_engine import TradingSignal

logger = logging.getLogger(__name__)


class SignalFilter:
    """Base class for signal filters"""

    def __init__(self, name: str):
        self.name = name
        self.stats = {"total_signals": 0, "passed_signals": 0, "filtered_signals": 0}

    def filter(self, signal: TradingSignal) -> bool:
        """Return True if signal passes filter, False otherwise"""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            "name": self.name,
            "total_signals": self.stats["total_signals"],
            "passed_signals": self.stats["passed_signals"],
            "filtered_signals": self.stats["filtered_signals"],
            "pass_rate": self.stats["passed_signals"] / max(1, self.stats["total_signals"]),
        }


class ConfidenceFilter(SignalFilter):
    """Filter signals based on confidence threshold"""

    def __init__(self, min_confidence: float = 0.5):
        super().__init__("ConfidenceFilter")
        self.min_confidence = min_confidence

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        if signal.confidence >= self.min_confidence:
            self.stats["passed_signals"] += 1
            return True
        else:
            self.stats["filtered_signals"] += 1
            logger.debug(
                f"Signal {signal.id} filtered: confidence {signal.confidence:.2f} < {self.min_confidence}"
            )
            return False


class QualityScoreFilter(SignalFilter):
    """Filter signals based on data quality score"""

    def __init__(self, min_quality_score: float = 0.7):
        super().__init__("QualityScoreFilter")
        self.min_quality_score = min_quality_score

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        if signal.quality_score >= self.min_quality_score:
            self.stats["passed_signals"] += 1
            return True
        else:
            self.stats["filtered_signals"] += 1
            logger.debug(
                f"Signal {signal.id} filtered: quality score {signal.quality_score:.2f} < {self.min_quality_score}"
            )
            return False


class RiskFilter(SignalFilter):
    """Filter signals based on risk level"""

    def __init__(self, allowed_risk_levels: List[str] = None):
        super().__init__("RiskFilter")
        self.allowed_risk_levels = allowed_risk_levels or ["low", "medium"]

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        if signal.risk_level in self.allowed_risk_levels:
            self.stats["passed_signals"] += 1
            return True
        else:
            self.stats["filtered_signals"] += 1
            logger.debug(
                f"Signal {signal.id} filtered: risk level {signal.risk_level} not in {self.allowed_risk_levels}"
            )
            return False


class VolumeFilter(SignalFilter):
    """Filter signals based on volume criteria"""

    def __init__(self, min_volume_ratio: float = 1.5):
        super().__init__("VolumeFilter")
        self.min_volume_ratio = min_volume_ratio

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        volume_ratio = signal.indicators.get("volume_ratio", 0)
        if volume_ratio >= self.min_volume_ratio:
            self.stats["passed_signals"] += 1
            return True
        else:
            self.stats["filtered_signals"] += 1
            logger.debug(
                f"Signal {signal.id} filtered: volume ratio {volume_ratio:.2f} < {self.min_volume_ratio}"
            )
            return False


class TechnicalConsistencyFilter(SignalFilter):
    """Filter signals based on technical indicator consistency"""

    def __init__(self, min_consistent_indicators: int = 2):
        super().__init__("TechnicalConsistencyFilter")
        self.min_consistent_indicators = min_consistent_indicators

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        # Count consistent indicators
        consistent_count = 0
        action = signal.action

        # RSI consistency
        rsi = signal.indicators.get("rsi", 50)
        if (action == "BUY" and rsi < 40) or (action == "SELL" and rsi > 60):
            consistent_count += 1

        # MACD consistency
        macd = signal.indicators.get("macd", 0)
        macd_signal = signal.indicators.get("macd_signal", 0)
        if (action == "BUY" and macd > macd_signal) or (action == "SELL" and macd < macd_signal):
            consistent_count += 1

        # Bollinger Bands consistency
        bb_percent = signal.indicators.get("bb_percent", 0.5)
        if (action == "BUY" and bb_percent < 0.3) or (action == "SELL" and bb_percent > 0.7):
            consistent_count += 1

        # Stochastic consistency
        stoch_k = signal.indicators.get("stoch_k", 50)
        if (action == "BUY" and stoch_k < 30) or (action == "SELL" and stoch_k > 70):
            consistent_count += 1

        if consistent_count >= self.min_consistent_indicators:
            self.stats["passed_signals"] += 1
            return True
        else:
            self.stats["filtered_signals"] += 1
            logger.debug(
                f"Signal {signal.id} filtered: only {consistent_count} consistent indicators"
            )
            return False


class DuplicateSignalFilter(SignalFilter):
    """Filter duplicate or similar signals within a time window"""

    def __init__(self, time_window_minutes: int = 30):
        super().__init__("DuplicateSignalFilter")
        self.time_window = timedelta(minutes=time_window_minutes)
        self.recent_signals = defaultdict(list)

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        # Clean old signals
        cutoff_time = datetime.now(timezone.utc) - self.time_window
        for symbol in list(self.recent_signals.keys()):
            self.recent_signals[symbol] = [
                s for s in self.recent_signals[symbol] if s.timestamp > cutoff_time
            ]

        # Check for duplicates
        symbol_signals = self.recent_signals[signal.symbol]
        for recent_signal in symbol_signals:
            if recent_signal.action == signal.action:
                # Check if price is similar (within 0.5%)
                price_diff = abs(recent_signal.price - signal.price) / signal.price
                if price_diff < 0.005:
                    self.stats["filtered_signals"] += 1
                    logger.debug(f"Signal {signal.id} filtered: duplicate of {recent_signal.id}")
                    return False

        # Add to recent signals
        self.recent_signals[signal.symbol].append(signal)
        self.stats["passed_signals"] += 1
        return True


class MarketConditionFilter(SignalFilter):
    """Filter signals based on overall market conditions"""

    def __init__(self, market_data_validator=None):
        super().__init__("MarketConditionFilter")
        self.market_data_validator = market_data_validator

    def filter(self, signal: TradingSignal) -> bool:
        self.stats["total_signals"] += 1

        # For now, always pass (can be enhanced with market breadth analysis)
        # Future enhancements:
        # - Check VIX levels
        # - Check market breadth
        # - Check sector rotation
        # - Check correlation with major indices

        self.stats["passed_signals"] += 1
        return True


class SignalFilteringPipeline:
    """
    Multi-stage signal filtering pipeline
    Ensures only high-quality, actionable signals pass through
    """

    def __init__(self):
        self.filters: List[SignalFilter] = []
        self.stats = {
            "total_signals_processed": 0,
            "signals_passed": 0,
            "signals_filtered": 0,
            "filter_stats": [],
        }

        # Initialize default filters
        self._initialize_default_filters()

    def _initialize_default_filters(self):
        """Initialize default filter pipeline"""
        self.add_filter(ConfidenceFilter(min_confidence=0.5))
        self.add_filter(QualityScoreFilter(min_quality_score=0.7))
        self.add_filter(RiskFilter(allowed_risk_levels=["low", "medium"]))
        self.add_filter(VolumeFilter(min_volume_ratio=1.2))
        self.add_filter(TechnicalConsistencyFilter(min_consistent_indicators=2))
        self.add_filter(DuplicateSignalFilter(time_window_minutes=30))
        self.add_filter(MarketConditionFilter())

    def add_filter(self, filter_obj: SignalFilter):
        """Add a filter to the pipeline"""
        self.filters.append(filter_obj)
        logger.info(f"Added filter: {filter_obj.name}")

    def remove_filter(self, filter_name: str):
        """Remove a filter from the pipeline"""
        self.filters = [f for f in self.filters if f.name != filter_name]
        logger.info(f"Removed filter: {filter_name}")

    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter a list of signals through the pipeline"""
        filtered_signals = []

        for signal in signals:
            self.stats["total_signals_processed"] += 1

            if self._passes_all_filters(signal):
                filtered_signals.append(signal)
                self.stats["signals_passed"] += 1
            else:
                self.stats["signals_filtered"] += 1

        logger.info(f"Filtered {len(signals)} signals -> {len(filtered_signals)} passed")
        return filtered_signals

    def _passes_all_filters(self, signal: TradingSignal) -> bool:
        """Check if signal passes all filters"""
        for filter_obj in self.filters:
            if not filter_obj.filter(signal):
                return False
        return True

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        self.stats["filter_stats"] = [f.get_stats() for f in self.filters]
        self.stats["overall_pass_rate"] = self.stats["signals_passed"] / max(
            1, self.stats["total_signals_processed"]
        )

        return self.stats

    def adjust_filter_parameters(self, performance_metrics: Dict[str, float]):
        """
        Dynamically adjust filter parameters based on performance
        This can be called periodically to optimize the pipeline
        """
        # Example adjustments based on performance
        if "false_positive_rate" in performance_metrics:
            fpr = performance_metrics["false_positive_rate"]

            # If too many false positives, increase confidence threshold
            if fpr > 0.3:
                for filter_obj in self.filters:
                    if isinstance(filter_obj, ConfidenceFilter):
                        filter_obj.min_confidence = min(0.8, filter_obj.min_confidence + 0.1)
                        logger.info(
                            f"Increased confidence threshold to {filter_obj.min_confidence}"
                        )

        if "signal_accuracy" in performance_metrics:
            accuracy = performance_metrics["signal_accuracy"]

            # If accuracy is low, increase quality requirements
            if accuracy < 0.5:
                for filter_obj in self.filters:
                    if isinstance(filter_obj, QualityScoreFilter):
                        filter_obj.min_quality_score = min(0.9, filter_obj.min_quality_score + 0.05)
                        logger.info(
                            f"Increased quality threshold to {filter_obj.min_quality_score}"
                        )

    def create_custom_pipeline(self, config: Dict[str, Any]) -> "SignalFilteringPipeline":
        """Create a custom pipeline with specific configuration"""
        custom_pipeline = SignalFilteringPipeline()
        custom_pipeline.filters = []  # Clear default filters

        # Add filters based on configuration
        if config.get("confidence_filter", True):
            custom_pipeline.add_filter(
                ConfidenceFilter(min_confidence=config.get("min_confidence", 0.5))
            )

        if config.get("quality_filter", True):
            custom_pipeline.add_filter(
                QualityScoreFilter(min_quality_score=config.get("min_quality_score", 0.7))
            )

        if config.get("risk_filter", True):
            custom_pipeline.add_filter(
                RiskFilter(allowed_risk_levels=config.get("allowed_risk_levels", ["low", "medium"]))
            )

        if config.get("volume_filter", True):
            custom_pipeline.add_filter(
                VolumeFilter(min_volume_ratio=config.get("min_volume_ratio", 1.2))
            )

        if config.get("technical_consistency_filter", True):
            custom_pipeline.add_filter(
                TechnicalConsistencyFilter(
                    min_consistent_indicators=config.get("min_consistent_indicators", 2)
                )
            )

        if config.get("duplicate_filter", True):
            custom_pipeline.add_filter(
                DuplicateSignalFilter(
                    time_window_minutes=config.get("duplicate_window_minutes", 30)
                )
            )

        return custom_pipeline
