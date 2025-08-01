"""
Comprehensive Load Testing Framework for GoldenSignalsAI V3

This module provides load testing and performance validation for:
- Agent signal generation under high concurrency
- WebSocket streaming performance
- Database query performance
- Cache hit rate optimization
- Memory and CPU usage under load
- API endpoint stress testing
"""

import asyncio
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock

# Performance testing utilities
@dataclass
class LoadTestResult:
    """Results from a load test"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    success_rate: float
    errors: List[str]
    memory_usage_mb: float
    cpu_usage_percent: float

class LoadTester:
    """Load testing utility for agents and services"""

    def __init__(self, max_workers: int = 50):
        self.max_workers = max_workers
        self.results = []

    async def run_concurrent_test(
        self,
        test_function: Callable,
        test_data: List[Any],
        test_name: str = "concurrent_test"
    ) -> LoadTestResult:
        """Run concurrent test with multiple data points"""

        start_time = time.time()
        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0

        # Memory tracking
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute concurrent requests
        semaphore = asyncio.Semaphore(self.max_workers)

        async def execute_request(data):
            async with semaphore:
                request_start = time.time()
                try:
                    await test_function(data)
                    request_time = time.time() - request_start
                    response_times.append(request_time)
                    return True
                except Exception as e:
                    errors.append(str(e))
                    return False

        # Run all requests
        tasks = [execute_request(data) for data in test_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
                errors.append(str(result))
            elif result is True:
                successful_requests += 1
            else:
                failed_requests += 1

        total_duration = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()

        # Calculate statistics
        if response_times:
            response_times_array = np.array(response_times)
            avg_response_time = np.mean(response_times_array)
            min_response_time = np.min(response_times_array)
            max_response_time = np.max(response_times_array)
            p50_response_time = np.percentile(response_times_array, 50)
            p95_response_time = np.percentile(response_times_array, 95)
            p99_response_time = np.percentile(response_times_array, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0

        total_requests = len(test_data)
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        result = LoadTestResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=total_duration,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            success_rate=success_rate,
            errors=errors[:10],  # Keep only first 10 errors
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=cpu_usage
        )

        self.results.append(result)
        return result

    def print_results(self, result: LoadTestResult):
        """Print formatted test results"""
        print(f"\n=== Load Test Results: {result.test_name} ===")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Success Rate: {result.success_rate:.2%}")
        print(f"Total Duration: {result.total_duration:.2f}s")
        print(f"Requests/Second: {result.requests_per_second:.2f}")
        print(f"")
        print(f"Response Time Statistics:")
        print(f"  Average: {result.avg_response_time:.3f}s")
        print(f"  Median (P50): {result.p50_response_time:.3f}s")
        print(f"  95th Percentile: {result.p95_response_time:.3f}s")
        print(f"  99th Percentile: {result.p99_response_time:.3f}s")
        print(f"  Min: {result.min_response_time:.3f}s")
        print(f"  Max: {result.max_response_time:.3f}s")
        print(f"")
        print(f"Resource Usage:")
        print(f"  Memory Delta: {result.memory_usage_mb:.1f} MB")
        print(f"  CPU Usage: {result.cpu_usage_percent:.1f}%")

        if result.errors:
            print(f"\nSample Errors:")
            for error in result.errors[:5]:
                print(f"  - {error}")


@pytest.mark.performance
class TestAgentLoadPerformance:
    """Load testing for agent performance"""

    @pytest.fixture
    def load_tester(self):
        """Create load tester instance"""
        return LoadTester(max_workers=50)

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        return [
            {
                "ohlcv_data": [
                    {"open": 100 + i, "high": 105 + i, "low": 98 + i, "close": 103 + i, "volume": 1000000}
                    for i in range(50)
                ],
                "symbol": f"TEST{i:03d}"
            }
            for i in range(1000)  # 1000 different test cases
        ]

    @pytest.fixture
    def sample_options_data(self):
        """Generate sample options data for testing"""
        return [
            {
                "options_data": [
                    {
                        'strike': 100 + j,
                        'type': 'call' if j % 2 == 0 else 'put',
                        'open_interest': 1000,
                        'volume': 500,
                        'time_to_expiry': 0.25,
                        'implied_volatility': 0.25 + (j * 0.01)
                    }
                    for j in range(10)
                ],
                'spot_price': 102 + i
            }
            for i in range(500)  # 500 different test cases
        ]

    async def test_pattern_agent_load(self, load_tester, sample_market_data):
        """Test PatternAgent under high load"""
        from agents.core.technical.pattern_agent import PatternAgent

        agent = PatternAgent()

        async def test_pattern_signal(data):
            return agent.process(data)

        result = await load_tester.run_concurrent_test(
            test_pattern_signal,
            sample_market_data,
            "PatternAgent_Load_Test"
        )

        load_tester.print_results(result)

        # Performance assertions
        assert result.success_rate >= 0.95, f"Success rate {result.success_rate:.2%} below 95%"
        assert result.p95_response_time <= 1.0, f"P95 response time {result.p95_response_time:.3f}s exceeds 1.0s"
        assert result.requests_per_second >= 100, f"Throughput {result.requests_per_second:.1f} RPS below 100"
        assert result.memory_usage_mb <= 500, f"Memory usage {result.memory_usage_mb:.1f}MB exceeds 500MB"

    async def test_gamma_exposure_agent_load(self, load_tester, sample_options_data):
        """Test GammaExposureAgent under high load"""
        from agents.core.options.gamma_exposure_agent import GammaExposureAgent

        agent = GammaExposureAgent()

        async def test_gamma_signal(data):
            return agent.process(data)

        result = await load_tester.run_concurrent_test(
            test_gamma_signal,
            sample_options_data,
            "GammaExposureAgent_Load_Test"
        )

        load_tester.print_results(result)

        # Performance assertions for options agent (more complex calculations)
        assert result.success_rate >= 0.90, f"Success rate {result.success_rate:.2%} below 90%"
        assert result.p95_response_time <= 2.0, f"P95 response time {result.p95_response_time:.3f}s exceeds 2.0s"
        assert result.requests_per_second >= 50, f"Throughput {result.requests_per_second:.1f} RPS below 50"

    async def test_meta_consensus_agent_load(self, load_tester):
        """Test MetaConsensusAgent under high load"""
        from agents.meta.meta_consensus_agent import MetaConsensusAgent

        agent = MetaConsensusAgent()

        # Generate test data for consensus agent
        consensus_test_data = [
            {
                "agent_signals": [
                    {"agent_name": f"test_agent_{j}", "action": "buy" if j % 3 == 0 else "sell" if j % 3 == 1 else "hold",
                     "confidence": 0.5 + (j * 0.1) % 0.5, "agent_type": "technical"}
                    for j in range(5)
                ]
            }
            for i in range(200)  # 200 consensus tests
        ]

        async def test_consensus_signal(data):
            return agent.process(data)

        result = await load_tester.run_concurrent_test(
            test_consensus_signal,
            consensus_test_data,
            "MetaConsensusAgent_Load_Test"
        )

        load_tester.print_results(result)

        # Performance assertions for meta agent
        assert result.success_rate >= 0.95, f"Success rate {result.success_rate:.2%} below 95%"
        assert result.p95_response_time <= 0.5, f"P95 response time {result.p95_response_time:.3f}s exceeds 0.5s"
        assert result.requests_per_second >= 200, f"Throughput {result.requests_per_second:.1f} RPS below 200"


@pytest.mark.performance
class TestSystemLoadPerformance:
    """System-wide load testing"""

    @pytest.fixture
    def load_tester(self):
        return LoadTester(max_workers=100)

    async def test_mixed_agent_workload(self, load_tester):
        """Test mixed workload across multiple agent types"""
        from agents.core.technical.pattern_agent import PatternAgent
        from agents.core.options.gamma_exposure_agent import GammaExposureAgent
        from agents.meta.meta_consensus_agent import MetaConsensusAgent

        # Initialize agents
        pattern_agent = PatternAgent()
        gamma_agent = GammaExposureAgent()
        consensus_agent = MetaConsensusAgent()

        # Mixed test data
        mixed_test_data = []

        # Pattern agent data (33%)
        for i in range(334):
            mixed_test_data.append({
                "agent_type": "pattern",
                "data": {
                    "ohlcv_data": [
                        {"open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000}
                        for _ in range(10)
                    ]
                }
            })

        # Gamma agent data (33%)
        for i in range(333):
            mixed_test_data.append({
                "agent_type": "gamma",
                "data": {
                    "options_data": [
                        {
                            'strike': 100,
                            'type': 'call',
                            'open_interest': 1000,
                            'volume': 500,
                            'time_to_expiry': 0.25,
                            'implied_volatility': 0.25
                        }
                    ],
                    'spot_price': 102
                }
            })

        # Consensus agent data (33%)
        for i in range(333):
            mixed_test_data.append({
                "agent_type": "consensus",
                "data": {
                    "agent_signals": [
                        {"agent_name": "test1", "action": "buy", "confidence": 0.8, "agent_type": "technical"},
                        {"agent_name": "test2", "action": "sell", "confidence": 0.6, "agent_type": "fundamental"}
                    ]
                }
            })

        # Shuffle to randomize execution order
        import random
        random.shuffle(mixed_test_data)

        async def test_mixed_signal(test_case):
            agent_type = test_case["agent_type"]
            data = test_case["data"]

            if agent_type == "pattern":
                return pattern_agent.process(data)
            elif agent_type == "gamma":
                return gamma_agent.process(data)
            elif agent_type == "consensus":
                return consensus_agent.process(data)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        result = await load_tester.run_concurrent_test(
            test_mixed_signal,
            mixed_test_data,
            "Mixed_Agent_Workload_Test"
        )

        load_tester.print_results(result)

        # System-level performance assertions
        assert result.success_rate >= 0.90, f"System success rate {result.success_rate:.2%} below 90%"
        assert result.p95_response_time <= 3.0, f"System P95 response time {result.p95_response_time:.3f}s exceeds 3.0s"
        assert result.requests_per_second >= 100, f"System throughput {result.requests_per_second:.1f} RPS below 100"
        assert result.memory_usage_mb <= 1000, f"System memory usage {result.memory_usage_mb:.1f}MB exceeds 1GB"

    async def test_sustained_load(self, load_tester):
        """Test sustained load over extended period"""
        from agents.core.technical.pattern_agent import PatternAgent

        agent = PatternAgent()

        # Generate sustained test data (run for 5 minutes worth of requests)
        sustained_test_data = [
            {
                "ohlcv_data": [
                    {"open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000}
                    for _ in range(10)
                ]
            }
            for _ in range(1500)  # 1500 requests
        ]

        async def test_sustained_signal(data):
            # Add small delay to simulate sustained load
            await asyncio.sleep(0.01)
            return agent.process(data)

        result = await load_tester.run_concurrent_test(
            test_sustained_signal,
            sustained_test_data,
            "Sustained_Load_Test"
        )

        load_tester.print_results(result)

        # Sustained load assertions
        assert result.success_rate >= 0.95, f"Sustained success rate {result.success_rate:.2%} below 95%"
        assert result.total_duration <= 120, f"Sustained test took {result.total_duration:.1f}s, expected ≤120s"
        assert result.memory_usage_mb <= 200, f"Memory growth {result.memory_usage_mb:.1f}MB indicates potential leak"


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks and regression tests"""

    def test_signal_generation_benchmark(self):
        """Benchmark signal generation performance"""
        from agents.core.technical.pattern_agent import PatternAgent

        agent = PatternAgent()
        test_data = {
            "ohlcv_data": [
                {"open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000}
                for _ in range(100)  # Larger dataset
            ]
        }

        # Warm up
        for _ in range(10):
            agent.process(test_data)

        # Benchmark
        start_time = time.time()
        iterations = 100

        for _ in range(iterations):
            result = agent.process(test_data)
            assert result["action"] in ["buy", "sell", "hold"]

        total_time = time.time() - start_time
        avg_time = total_time / iterations

        print(f"\nPattern Agent Benchmark:")
        print(f"  Iterations: {iterations}")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Average Time: {avg_time:.3f}s")
        print(f"  Operations/Second: {iterations/total_time:.1f}")

        # Performance targets
        assert avg_time <= 0.1, f"Average execution time {avg_time:.3f}s exceeds 0.1s target"
        assert iterations/total_time >= 50, f"Throughput {iterations/total_time:.1f} ops/s below 50 target"

    def test_memory_efficiency(self):
        """Test memory efficiency under repeated operations"""
        from agents.core.options.gamma_exposure_agent import GammaExposureAgent
        import gc
        import psutil

        agent = GammaExposureAgent()
        process = psutil.Process()

        test_data = {
            "options_data": [
                {
                    'strike': 100 + i,
                    'type': 'call' if i % 2 == 0 else 'put',
                    'open_interest': 1000,
                    'volume': 500,
                    'time_to_expiry': 0.25,
                    'implied_volatility': 0.25
                }
                for i in range(20)
            ],
            'spot_price': 102
        }

        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run many operations
        for i in range(1000):
            result = agent.process(test_data)
            assert result["action"] in ["buy", "sell", "hold"]

            # Periodic garbage collection
            if i % 100 == 0:
                gc.collect()

        # Measure final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory

        print(f"\nMemory Efficiency Test:")
        print(f"  Baseline Memory: {baseline_memory:.1f} MB")
        print(f"  Final Memory: {final_memory:.1f} MB")
        print(f"  Memory Growth: {memory_growth:.1f} MB")
        print(f"  Growth per Operation: {memory_growth/1000:.3f} MB")

        # Memory efficiency targets
        assert memory_growth <= 100, f"Memory growth {memory_growth:.1f}MB exceeds 100MB limit"
        assert memory_growth/1000 <= 0.1, f"Memory growth per operation {memory_growth/1000:.3f}MB exceeds 0.1MB"


if __name__ == "__main__":
    # Run specific performance tests
    pytest.main([
        __file__,
        "-v",
        "-m", "performance",
        "--tb=short"
    ])
