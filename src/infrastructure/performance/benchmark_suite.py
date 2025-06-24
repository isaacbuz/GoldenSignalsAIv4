"""
Performance Benchmark Suite for GoldenSignalsAI V2
Issue #197: Performance Optimization - Performance Benchmarks
"""

import asyncio
import gc
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    name: str
    category: str
    iterations: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    memory_start: float
    memory_peak: float
    memory_end: float
    cpu_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def ops_per_second(self) -> float:
        """Calculate operations per second"""
        return self.iterations / self.total_time if self.total_time > 0 else 0
    
    @property
    def memory_growth(self) -> float:
        """Calculate memory growth in MB"""
        return self.memory_end - self.memory_start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "category": self.category,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "std_dev": self.std_dev,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "ops_per_second": self.ops_per_second,
            "memory_start_mb": self.memory_start,
            "memory_peak_mb": self.memory_peak,
            "memory_end_mb": self.memory_end,
            "memory_growth_mb": self.memory_growth,
            "cpu_percent": self.cpu_percent,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class PerformanceBenchmark:
    """Base performance benchmark class"""
    
    def __init__(
        self,
        name: str,
        category: str,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        enable_profiling: bool = True
    ):
        self.name = name
        self.category = category
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.enable_profiling = enable_profiling
        
        # Timing results
        self.times: List[float] = []
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        
        # Process info
        self.process = psutil.Process()
        
    async def setup(self):
        """Setup benchmark environment"""
        pass
    
    async def teardown(self):
        """Cleanup after benchmark"""
        pass
    
    async def run_iteration(self) -> Any:
        """Run single benchmark iteration - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _collect_memory_stats(self) -> Tuple[float, float, float]:
        """Collect memory statistics"""
        gc.collect()
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def _collect_cpu_stats(self) -> float:
        """Collect CPU statistics"""
        return self.process.cpu_percent(interval=0.1)
    
    async def run(self) -> BenchmarkResult:
        """Run the benchmark"""
        logger.info(f"Starting benchmark: {self.name}")
        
        # Setup
        await self.setup()
        
        try:
            # Warmup runs
            logger.info(f"Running {self.warmup_iterations} warmup iterations")
            for _ in range(self.warmup_iterations):
                await self.run_iteration()
            
            # Clear any caches and collect garbage
            gc.collect()
            
            # Record initial memory
            memory_start = self._collect_memory_stats()
            memory_peak = memory_start
            
            # Benchmark runs
            logger.info(f"Running {self.benchmark_iterations} benchmark iterations")
            
            for i in range(self.benchmark_iterations):
                # Time the iteration
                start_time = time.perf_counter()
                await self.run_iteration()
                end_time = time.perf_counter()
                
                iteration_time = end_time - start_time
                self.times.append(iteration_time)
                
                # Collect resource stats periodically
                if i % 10 == 0:
                    current_memory = self._collect_memory_stats()
                    memory_peak = max(memory_peak, current_memory)
                    self.memory_samples.append(current_memory)
                    
                    if self.enable_profiling:
                        cpu_percent = self._collect_cpu_stats()
                        self.cpu_samples.append(cpu_percent)
            
            # Final memory reading
            memory_end = self._collect_memory_stats()
            
            # Calculate statistics
            result = BenchmarkResult(
                name=self.name,
                category=self.category,
                iterations=self.benchmark_iterations,
                total_time=sum(self.times),
                min_time=min(self.times),
                max_time=max(self.times),
                avg_time=statistics.mean(self.times),
                median_time=statistics.median(self.times),
                std_dev=statistics.stdev(self.times) if len(self.times) > 1 else 0,
                percentile_95=np.percentile(self.times, 95),
                percentile_99=np.percentile(self.times, 99),
                memory_start=memory_start,
                memory_peak=memory_peak,
                memory_end=memory_end,
                cpu_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0
            )
            
            logger.info(f"Benchmark complete: {self.name} - {result.ops_per_second:.2f} ops/sec")
            
            return result
            
        finally:
            await self.teardown()


class BenchmarkSuite:
    """Suite of performance benchmarks"""
    
    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.benchmarks: List[PerformanceBenchmark] = []
        self.results: List[BenchmarkResult] = []
        
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add benchmark to suite"""
        self.benchmarks.append(benchmark)
    
    async def run_all(self, save_results: bool = True) -> List[BenchmarkResult]:
        """Run all benchmarks in suite"""
        logger.info(f"Starting benchmark suite: {self.suite_name}")
        
        for benchmark in self.benchmarks:
            try:
                result = await benchmark.run()
                self.results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark.name} failed: {e}")
        
        if save_results:
            self._save_results()
        
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{self.suite_name}_{timestamp}.json"
        
        results_data = {
            "suite_name": self.suite_name,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def _print_summary(self):
        """Print summary of results"""
        print(f"\n{'=' * 80}")
        print(f"Benchmark Suite: {self.suite_name}")
        print(f"{'=' * 80}")
        print(f"{'Name':<30} {'Category':<15} {'Ops/sec':>12} {'Avg Time (ms)':>15} {'Memory (MB)':>12}")
        print(f"{'-' * 80}")
        
        for result in self.results:
            print(f"{result.name:<30} {result.category:<15} "
                  f"{result.ops_per_second:>12.2f} "
                  f"{result.avg_time * 1000:>15.3f} "
                  f"{result.memory_growth:>12.2f}")
        
        print(f"{'=' * 80}\n")


# Specific benchmark implementations

class CacheBenchmark(PerformanceBenchmark):
    """Benchmark cache operations"""
    
    def __init__(self, cache_manager, data_size: int = 1000):
        super().__init__(
            name=f"Cache_Operations_{data_size}",
            category="Cache",
            benchmark_iterations=10000
        )
        self.cache = cache_manager
        self.data_size = data_size
        self.test_data = {}
        
    async def setup(self):
        """Generate test data"""
        self.test_data = {
            f"key_{i}": {"value": f"data_{i}" * 100, "timestamp": time.time()}
            for i in range(self.data_size)
        }
        
        # Pre-populate cache
        for key, value in list(self.test_data.items())[:self.data_size // 2]:
            await self.cache.set("benchmark", key, value)
    
    async def run_iteration(self):
        """Run cache operations"""
        key = f"key_{np.random.randint(0, self.data_size)}"
        
        # 70% reads, 20% writes, 10% deletes
        operation = np.random.choice(["read", "write", "delete"], p=[0.7, 0.2, 0.1])
        
        if operation == "read":
            await self.cache.get("benchmark", key)
        elif operation == "write":
            await self.cache.set("benchmark", key, self.test_data[key])
        else:
            await self.cache.delete("benchmark", key)


class DatabaseBenchmark(PerformanceBenchmark):
    """Benchmark database operations"""
    
    def __init__(self, db_optimizer, query_type: str = "simple"):
        super().__init__(
            name=f"Database_{query_type}",
            category="Database",
            benchmark_iterations=1000
        )
        self.db = db_optimizer
        self.query_type = query_type
        
    async def setup(self):
        """Setup test tables"""
        # This would create test tables in a real implementation
        pass
    
    async def run_iteration(self):
        """Run database query"""
        if self.query_type == "simple":
            query = "SELECT * FROM signals WHERE symbol = $1 LIMIT 10"
            params = ("AAPL",)
        elif self.query_type == "complex":
            query = """
                SELECT s.*, p.total_value 
                FROM signals s 
                JOIN portfolio p ON s.symbol = p.symbol 
                WHERE s.created_at > NOW() - INTERVAL '1 hour' 
                ORDER BY s.created_at DESC 
                LIMIT 100
            """
            params = None
        elif self.query_type == "aggregate":
            query = """
                SELECT symbol, 
                       COUNT(*) as signal_count,
                       AVG(confidence) as avg_confidence
                FROM signals 
                WHERE created_at > $1
                GROUP BY symbol
                HAVING COUNT(*) > 10
            """
            params = (datetime.now(),)
        else:
            query = "SELECT 1"
            params = None
        
        await self.db.execute_query(query, params)


class DataProcessingBenchmark(PerformanceBenchmark):
    """Benchmark data processing operations"""
    
    def __init__(self, data_size: int = 10000):
        super().__init__(
            name=f"Data_Processing_{data_size}",
            category="Processing",
            benchmark_iterations=100
        )
        self.data_size = data_size
        self.df = None
        
    async def setup(self):
        """Generate test dataframe"""
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=self.data_size, freq="1min"),
            "price": np.random.uniform(100, 200, self.data_size),
            "volume": np.random.randint(1000, 100000, self.data_size),
            "signal": np.random.choice(["BUY", "SELL", "HOLD"], self.data_size)
        })
    
    async def run_iteration(self):
        """Process data"""
        # Calculate moving averages
        self.df["ma_20"] = self.df["price"].rolling(20).mean()
        self.df["ma_50"] = self.df["price"].rolling(50).mean()
        
        # Calculate RSI
        delta = self.df["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.df["rsi"] = 100 - (100 / (1 + rs))
        
        # Group and aggregate
        summary = self.df.groupby("signal").agg({
            "price": ["mean", "std"],
            "volume": "sum",
            "rsi": "mean"
        })
        
        return summary


class ConcurrencyBenchmark(PerformanceBenchmark):
    """Benchmark concurrent operations"""
    
    def __init__(self, concurrency_level: int = 100):
        super().__init__(
            name=f"Concurrency_{concurrency_level}",
            category="Concurrency",
            benchmark_iterations=100
        )
        self.concurrency_level = concurrency_level
        
    async def run_iteration(self):
        """Run concurrent tasks"""
        async def task(task_id: int):
            # Simulate work
            await asyncio.sleep(0.001)
            return task_id * 2
        
        # Run tasks concurrently
        tasks = [task(i) for i in range(self.concurrency_level)]
        results = await asyncio.gather(*tasks)
        
        return sum(results)


class NetworkBenchmark(PerformanceBenchmark):
    """Benchmark network/API operations"""
    
    def __init__(self, endpoint_type: str = "rest"):
        super().__init__(
            name=f"Network_{endpoint_type}",
            category="Network",
            benchmark_iterations=500
        )
        self.endpoint_type = endpoint_type
        
    async def run_iteration(self):
        """Simulate network call"""
        # In real implementation, this would make actual network calls
        # For now, simulate with sleep
        latency = np.random.uniform(0.01, 0.05)  # 10-50ms
        await asyncio.sleep(latency)
        
        return {"status": "success", "latency": latency}


def create_comprehensive_benchmark_suite() -> BenchmarkSuite:
    """Create comprehensive benchmark suite for GoldenSignalsAI"""
    suite = BenchmarkSuite("GoldenSignalsAI_Performance")
    
    # Add cache benchmarks
    for size in [100, 1000, 10000]:
        # suite.add_benchmark(CacheBenchmark(cache_manager, size))
        pass
    
    # Add database benchmarks
    for query_type in ["simple", "complex", "aggregate"]:
        # suite.add_benchmark(DatabaseBenchmark(db_optimizer, query_type))
        pass
    
    # Add data processing benchmarks
    for size in [1000, 10000, 100000]:
        suite.add_benchmark(DataProcessingBenchmark(size))
    
    # Add concurrency benchmarks
    for level in [10, 100, 1000]:
        suite.add_benchmark(ConcurrencyBenchmark(level))
    
    # Add network benchmarks
    for endpoint in ["rest", "websocket", "grpc"]:
        suite.add_benchmark(NetworkBenchmark(endpoint))
    
    return suite


async def run_performance_benchmarks():
    """Run all performance benchmarks"""
    suite = create_comprehensive_benchmark_suite()
    results = await suite.run_all()
    
    # Generate performance report
    report = generate_performance_report(results)
    
    return report


def generate_performance_report(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Generate comprehensive performance report"""
    report = {
        "summary": {
            "total_benchmarks": len(results),
            "categories": list(set(r.category for r in results)),
            "timestamp": datetime.now().isoformat()
        },
        "by_category": {},
        "top_performers": [],
        "bottlenecks": [],
        "recommendations": []
    }
    
    # Group by category
    for category in report["summary"]["categories"]:
        category_results = [r for r in results if r.category == category]
        report["by_category"][category] = {
            "count": len(category_results),
            "avg_ops_per_sec": statistics.mean(r.ops_per_second for r in category_results),
            "total_memory_mb": sum(r.memory_growth for r in category_results)
        }
    
    # Find top performers
    sorted_by_ops = sorted(results, key=lambda r: r.ops_per_second, reverse=True)
    report["top_performers"] = [
        {"name": r.name, "ops_per_sec": r.ops_per_second}
        for r in sorted_by_ops[:5]
    ]
    
    # Find bottlenecks
    sorted_by_time = sorted(results, key=lambda r: r.avg_time, reverse=True)
    report["bottlenecks"] = [
        {"name": r.name, "avg_time_ms": r.avg_time * 1000}
        for r in sorted_by_time[:5]
    ]
    
    # Generate recommendations
    for result in results:
        if result.memory_growth > 100:  # More than 100MB growth
            report["recommendations"].append(
                f"{result.name}: High memory growth ({result.memory_growth:.1f}MB) - consider optimization"
            )
        
        if result.cpu_percent > 80:
            report["recommendations"].append(
                f"{result.name}: High CPU usage ({result.cpu_percent:.1f}%) - consider parallelization"
            )
        
        if result.std_dev / result.avg_time > 0.5:  # High variance
            report["recommendations"].append(
                f"{result.name}: High performance variance - investigate inconsistency"
            )
    
    return report 