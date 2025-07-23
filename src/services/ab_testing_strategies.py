"""
A/B Testing Service for Trading Strategies
Enables systematic testing and comparison of different trading approaches
"""

import asyncio
import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status"""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class AllocationMethod(Enum):
    """Traffic allocation methods"""

    RANDOM = "random"  # Pure random allocation
    DETERMINISTIC = "deterministic"  # Hash-based deterministic
    WEIGHTED = "weighted"  # Weighted random allocation


@dataclass
class StrategyVariant:
    """Represents a strategy variant in A/B test"""

    id: str
    name: str
    description: str
    config: Dict[str, Any]  # Strategy configuration
    allocation_percentage: float
    metrics: Dict[str, float] = field(default_factory=dict)
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_return: float = 0.0
    avg_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTest:
    """A/B test configuration and results"""

    id: str
    name: str
    description: str
    symbol: Optional[str]  # None for all symbols
    control_variant: StrategyVariant
    test_variants: List[StrategyVariant]
    status: TestStatus
    allocation_method: AllocationMethod
    min_sample_size: int
    confidence_level: float  # Statistical confidence (e.g., 0.95)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingService:
    """
    Service for managing A/B tests on trading strategies
    """

    def __init__(self):
        # Active tests storage
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: List[ABTest] = []

        # Allocation cache for deterministic routing
        self.allocation_cache: Dict[str, str] = {}

        # Metrics aggregation
        self.metrics_buffer: Dict[str, List[Dict[str, Any]]] = {}

        # Configuration
        self.max_concurrent_tests = 5
        self.default_min_sample_size = 100
        self.default_confidence_level = 0.95

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background tasks for metrics aggregation"""
        asyncio.create_task(self._metrics_aggregator())
        asyncio.create_task(self._test_monitor())

    async def create_test(
        self,
        name: str,
        description: str,
        control_config: Dict[str, Any],
        test_configs: List[Dict[str, Any]],
        symbol: Optional[str] = None,
        allocation_percentages: Optional[List[float]] = None,
        min_sample_size: Optional[int] = None,
        allocation_method: AllocationMethod = AllocationMethod.RANDOM,
    ) -> ABTest:
        """Create a new A/B test"""

        # Validate allocation percentages
        if allocation_percentages:
            total = sum(allocation_percentages) + (100 - sum(allocation_percentages))
            if abs(total - 100) > 0.01:
                raise ValueError("Allocation percentages must sum to 100")
        else:
            # Equal allocation
            num_variants = len(test_configs) + 1
            allocation_percentages = [100 / num_variants] * num_variants

        # Create test ID
        test_id = f"test_{uuid.uuid4().hex[:8]}"

        # Create control variant
        control_variant = StrategyVariant(
            id=f"{test_id}_control",
            name="Control",
            description="Control strategy",
            config=control_config,
            allocation_percentage=allocation_percentages[0],
        )

        # Create test variants
        test_variants = []
        for i, (config, allocation) in enumerate(zip(test_configs, allocation_percentages[1:])):
            variant = StrategyVariant(
                id=f"{test_id}_variant_{i+1}",
                name=f"Variant {i+1}",
                description=config.get("description", f"Test variant {i+1}"),
                config=config,
                allocation_percentage=allocation,
            )
            test_variants.append(variant)

        # Create test
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            symbol=symbol,
            control_variant=control_variant,
            test_variants=test_variants,
            status=TestStatus.DRAFT,
            allocation_method=allocation_method,
            min_sample_size=min_sample_size or self.default_min_sample_size,
            confidence_level=self.default_confidence_level,
        )

        # Store test
        self.active_tests[test_id] = test

        logger.info(f"Created A/B test: {test_id} - {name}")
        return test

    async def start_test(self, test_id: str) -> ABTest:
        """Start an A/B test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        if test.status != TestStatus.DRAFT:
            raise ValueError(f"Test {test_id} is not in DRAFT status")

        # Check concurrent test limit
        running_tests = sum(1 for t in self.active_tests.values() if t.status == TestStatus.RUNNING)
        if running_tests >= self.max_concurrent_tests:
            raise ValueError(f"Maximum concurrent tests ({self.max_concurrent_tests}) reached")

        # Start test
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now()

        logger.info(f"Started A/B test: {test_id}")
        return test

    async def stop_test(self, test_id: str, abort: bool = False) -> ABTest:
        """Stop an A/B test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        # Stop test
        test.status = TestStatus.ABORTED if abort else TestStatus.COMPLETED
        test.end_time = datetime.now()

        # Calculate results if not aborted
        if not abort:
            test.results = await self._calculate_test_results(test)

        # Move to completed tests
        self.completed_tests.append(test)
        del self.active_tests[test_id]

        logger.info(f"Stopped A/B test: {test_id} - {'Aborted' if abort else 'Completed'}")
        return test

    async def allocate_variant(
        self, symbol: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Allocate a variant for a given symbol/context
        Returns (variant_id, strategy_config)
        """
        # Find applicable tests
        applicable_tests = [
            test
            for test in self.active_tests.values()
            if test.status == TestStatus.RUNNING and (test.symbol is None or test.symbol == symbol)
        ]

        if not applicable_tests:
            # No active tests, return default strategy
            return "default", self._get_default_strategy()

        # For now, use the first applicable test
        # In production, you might want more sophisticated logic
        test = applicable_tests[0]

        # Allocate variant based on method
        if test.allocation_method == AllocationMethod.RANDOM:
            variant = self._random_allocation(test)
        elif test.allocation_method == AllocationMethod.DETERMINISTIC:
            variant = self._deterministic_allocation(test, symbol)
        else:  # WEIGHTED
            variant = self._weighted_allocation(test)

        return variant.id, variant.config

    async def record_trade_result(
        self,
        variant_id: str,
        symbol: str,
        decision: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ):
        """Record the result of a trade for a variant"""
        # Find the test and variant
        test = None
        variant = None

        for t in self.active_tests.values():
            if t.control_variant.id == variant_id:
                test = t
                variant = t.control_variant
                break
            for v in t.test_variants:
                if v.id == variant_id:
                    test = t
                    variant = v
                    break
            if test:
                break

        if not test or not variant:
            logger.warning(f"Variant {variant_id} not found in active tests")
            return

        # Update variant metrics
        variant.trades_count += 1

        if outcome:
            if outcome.get("profitable", False):
                variant.wins += 1
            else:
                variant.losses += 1

            variant.total_return += outcome.get("return_pct", 0)

        # Update average confidence
        confidence = decision.get("confidence", 0)
        variant.avg_confidence = (
            variant.avg_confidence * (variant.trades_count - 1) + confidence
        ) / variant.trades_count

        # Buffer for batch processing
        if test.id not in self.metrics_buffer:
            self.metrics_buffer[test.id] = []

        self.metrics_buffer[test.id].append(
            {
                "variant_id": variant_id,
                "symbol": symbol,
                "decision": decision,
                "outcome": outcome,
                "timestamp": datetime.now(),
            }
        )

    def _random_allocation(self, test: ABTest) -> StrategyVariant:
        """Random variant allocation"""
        rand = random.random() * 100
        cumulative = 0

        # Check control
        cumulative += test.control_variant.allocation_percentage
        if rand < cumulative:
            return test.control_variant

        # Check test variants
        for variant in test.test_variants:
            cumulative += variant.allocation_percentage
            if rand < cumulative:
                return variant

        # Fallback to last variant
        return test.test_variants[-1]

    def _deterministic_allocation(self, test: ABTest, symbol: str) -> StrategyVariant:
        """Deterministic allocation based on symbol hash"""
        # Check cache
        cache_key = f"{test.id}:{symbol}"
        if cache_key in self.allocation_cache:
            variant_id = self.allocation_cache[cache_key]
            if variant_id == test.control_variant.id:
                return test.control_variant
            for v in test.test_variants:
                if v.id == variant_id:
                    return v

        # Hash-based allocation
        hash_value = hash(symbol) % 100
        cumulative = 0

        # Check control
        cumulative += test.control_variant.allocation_percentage
        if hash_value < cumulative:
            self.allocation_cache[cache_key] = test.control_variant.id
            return test.control_variant

        # Check test variants
        for variant in test.test_variants:
            cumulative += variant.allocation_percentage
            if hash_value < cumulative:
                self.allocation_cache[cache_key] = variant.id
                return variant

        # Fallback
        return test.test_variants[-1]

    def _weighted_allocation(self, test: ABTest) -> StrategyVariant:
        """Weighted allocation based on current performance"""
        # For early stage, use random allocation
        total_trades = test.control_variant.trades_count + sum(
            v.trades_count for v in test.test_variants
        )

        if total_trades < 20:  # Not enough data
            return self._random_allocation(test)

        # Calculate performance-based weights
        variants = [test.control_variant] + test.test_variants
        weights = []

        for variant in variants:
            if variant.trades_count > 0:
                win_rate = variant.wins / variant.trades_count
                avg_return = variant.total_return / variant.trades_count
                # Combine win rate and return for weight
                weight = win_rate * 0.5 + (avg_return + 1) * 0.5
            else:
                weight = 0.5  # Neutral weight

            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(variants)] * len(variants)

        # Select based on weights
        return np.random.choice(variants, p=weights)

    async def _calculate_test_results(self, test: ABTest) -> Dict[str, Any]:
        """Calculate statistical results of the test"""
        variants = [test.control_variant] + test.test_variants

        # Basic metrics
        results = {
            "test_id": test.id,
            "duration_hours": (
                (test.end_time - test.start_time).total_seconds() / 3600
                if test.start_time and test.end_time
                else 0
            ),
            "total_trades": sum(v.trades_count for v in variants),
            "variants": {},
        }

        # Per-variant metrics
        for variant in variants:
            if variant.trades_count > 0:
                win_rate = variant.wins / variant.trades_count
                avg_return = variant.total_return / variant.trades_count

                results["variants"][variant.id] = {
                    "name": variant.name,
                    "trades_count": variant.trades_count,
                    "win_rate": win_rate,
                    "avg_return": avg_return,
                    "total_return": variant.total_return,
                    "avg_confidence": variant.avg_confidence,
                    "sharpe_ratio": self._calculate_sharpe(variant),
                }
            else:
                results["variants"][variant.id] = {
                    "name": variant.name,
                    "trades_count": 0,
                    "insufficient_data": True,
                }

        # Statistical significance testing
        if test.control_variant.trades_count >= 20 and all(
            v.trades_count >= 20 for v in test.test_variants
        ):
            # Perform t-tests
            control_returns = self._get_variant_returns(test.control_variant)

            for variant in test.test_variants:
                variant_returns = self._get_variant_returns(variant)

                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(control_returns, variant_returns)

                results["variants"][variant.id]["t_statistic"] = t_stat
                results["variants"][variant.id]["p_value"] = p_value
                results["variants"][variant.id]["significant"] = p_value < (
                    1 - test.confidence_level
                )

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.std(control_returns) ** 2 + np.std(variant_returns) ** 2) / 2
                )
                if pooled_std > 0:
                    effect_size = (np.mean(variant_returns) - np.mean(control_returns)) / pooled_std
                    results["variants"][variant.id]["effect_size"] = effect_size

        # Winner determination
        best_variant = max(
            variants,
            key=lambda v: v.total_return / v.trades_count if v.trades_count > 0 else -float("inf"),
        )

        results["winner"] = {
            "variant_id": best_variant.id,
            "name": best_variant.name,
            "improvement_pct": (
                (
                    (best_variant.total_return / best_variant.trades_count)
                    - (test.control_variant.total_return / test.control_variant.trades_count)
                )
                * 100
                if test.control_variant.trades_count > 0 and best_variant.trades_count > 0
                else 0
            ),
        }

        return results

    def _calculate_sharpe(self, variant: StrategyVariant) -> float:
        """Calculate Sharpe ratio for a variant"""
        if variant.trades_count < 2:
            return 0.0

        # This is simplified - in production you'd use actual return series
        # Assuming risk-free rate of 2% annually
        risk_free_rate = 0.02 / 252  # Daily
        avg_return = variant.total_return / variant.trades_count

        # Estimate volatility (simplified)
        if variant.wins + variant.losses > 0:
            win_rate = variant.wins / (variant.wins + variant.losses)
            # Approximate standard deviation
            volatility = np.sqrt(win_rate * (1 - win_rate))
        else:
            volatility = 1.0

        if volatility > 0:
            sharpe = (avg_return - risk_free_rate) / volatility
        else:
            sharpe = 0.0

        return sharpe

    def _get_variant_returns(self, variant: StrategyVariant) -> List[float]:
        """Get return series for a variant (simplified)"""
        # In production, you'd store actual trade returns
        # Here we simulate based on aggregated metrics
        returns = []

        for _ in range(variant.wins):
            # Simulate winning trades
            returns.append(abs(np.random.normal(0.02, 0.01)))

        for _ in range(variant.losses):
            # Simulate losing trades
            returns.append(-abs(np.random.normal(0.01, 0.005)))

        return returns

    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default strategy configuration"""
        return {
            "name": "default",
            "agent_weights": {
                "rsi_agent": 1.0,
                "macd_agent": 1.0,
                "sentiment_agent": 1.0,
                "volume_agent": 1.0,
                "momentum_agent": 1.0,
            },
            "risk_tolerance": 0.5,
            "position_sizing": "kelly",
            "max_position_pct": 0.1,
        }

    async def _metrics_aggregator(self):
        """Background task to aggregate metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Aggregate every minute

                # Process buffered metrics
                for test_id, metrics in self.metrics_buffer.items():
                    if metrics and test_id in self.active_tests:
                        # Aggregate and clear buffer
                        logger.info(f"Aggregating {len(metrics)} metrics for test {test_id}")
                        self.metrics_buffer[test_id] = []

            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")

    async def _test_monitor(self):
        """Monitor tests for completion criteria"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                for test in list(self.active_tests.values()):
                    if test.status != TestStatus.RUNNING:
                        continue

                    # Check if minimum sample size reached
                    total_trades = test.control_variant.trades_count + sum(
                        v.trades_count for v in test.test_variants
                    )

                    if total_trades >= test.min_sample_size * len(test.test_variants):
                        # Check for statistical significance
                        temp_results = await self._calculate_test_results(test)

                        # If we have significant results, consider auto-completing
                        significant_results = any(
                            v.get("significant", False) for v in temp_results["variants"].values()
                        )

                        if significant_results:
                            logger.info(f"Test {test.id} reached significance")
                            # Could auto-complete or notify

            except Exception as e:
                logger.error(f"Test monitor error: {e}")

    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get summary of active tests"""
        return [
            {
                "id": test.id,
                "name": test.name,
                "symbol": test.symbol,
                "status": test.status.value,
                "variants_count": len(test.test_variants) + 1,
                "total_trades": sum(
                    v.trades_count for v in [test.control_variant] + test.test_variants
                ),
                "start_time": test.start_time.isoformat() if test.start_time else None,
            }
            for test in self.active_tests.values()
        ]

    def get_test_details(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed test information"""
        test = self.active_tests.get(test_id)
        if not test:
            # Check completed tests
            test = next((t for t in self.completed_tests if t.id == test_id), None)

        if not test:
            return None

        return {
            "test": asdict(test),
            "current_results": asyncio.run(self._calculate_test_results(test))
            if test.status == TestStatus.RUNNING
            else test.results,
        }


# Singleton instance
ab_testing_service = ABTestingService()


# Convenience functions
async def create_strategy_test(
    name: str,
    control_config: Dict[str, Any],
    test_configs: List[Dict[str, Any]],
    symbol: Optional[str] = None,
) -> str:
    """Create a new A/B test for trading strategies"""
    test = await ab_testing_service.create_test(
        name=name,
        description=f"A/B test for {symbol or 'all symbols'}",
        control_config=control_config,
        test_configs=test_configs,
        symbol=symbol,
    )
    return test.id


async def get_strategy_variant(symbol: str) -> Tuple[str, Dict[str, Any]]:
    """Get the strategy variant to use for a symbol"""
    return await ab_testing_service.allocate_variant(symbol)


async def record_trade_outcome(
    variant_id: str, symbol: str, decision: Dict[str, Any], outcome: Dict[str, Any]
):
    """Record the outcome of a trade"""
    await ab_testing_service.record_trade_result(
        variant_id=variant_id, symbol=symbol, decision=decision, outcome=outcome
    )
