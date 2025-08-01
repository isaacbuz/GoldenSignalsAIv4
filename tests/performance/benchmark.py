"""Performance benchmarks for GoldenSignalsAI."""

import asyncio
import time
import statistics
from typing import List
import aiohttp
import pandas as pd

class PerformanceBenchmark:
    """Run performance benchmarks."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []

    async def benchmark_endpoint(self, endpoint: str, method: str = "GET", iterations: int = 100):
        """Benchmark a single endpoint."""
        times = []
        errors = 0

        async with aiohttp.ClientSession() as session:
            for _ in range(iterations):
                start = time.time()
                try:
                    async with session.request(method, f"{self.base_url}{endpoint}") as response:
                        await response.text()
                        if response.status >= 400:
                            errors += 1
                except Exception:
                    errors += 1

                times.append(time.time() - start)

        return {
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
            "errors": errors,
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "p95_time": statistics.quantiles(times, n=20)[18],  # 95th percentile
            "p99_time": statistics.quantiles(times, n=100)[98],  # 99th percentile
        }

    async def run_benchmarks(self):
        """Run all benchmarks."""
        endpoints = [
            ("/api/v1/health/", "GET"),
            ("/api/v1/signals/AAPL", "GET"),
            ("/api/v1/signals/latest", "GET"),
            ("/api/v1/portfolio/status", "GET"),
        ]

        for endpoint, method in endpoints:
            result = await self.benchmark_endpoint(endpoint, method)
            self.results.append(result)
            print(f"✅ Benchmarked {endpoint}: avg={result['avg_time']:.3f}s, p95={result['p95_time']:.3f}s")

    def save_results(self, filename: str = "benchmark_results.csv"):
        """Save benchmark results."""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"✅ Saved benchmark results to {filename}")

    def check_sla(self):
        """Check if results meet SLA requirements."""
        sla_requirements = {
            "/api/v1/health/": 0.1,  # 100ms
            "/api/v1/signals/": 0.5,  # 500ms
            "/api/v1/portfolio/": 1.0,  # 1s
        }

        violations = []
        for result in self.results:
            for endpoint_prefix, max_time in sla_requirements.items():
                if result["endpoint"].startswith(endpoint_prefix):
                    if result["p95_time"] > max_time:
                        violations.append(f"{result['endpoint']}: p95={result['p95_time']:.3f}s > SLA={max_time}s")

        if violations:
            print("❌ SLA Violations:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("✅ All endpoints meet SLA requirements")

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    asyncio.run(benchmark.run_benchmarks())
    benchmark.save_results()
    benchmark.check_sla()
