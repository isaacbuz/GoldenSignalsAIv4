"""
Locust load testing for GoldenSignalsAI API
"""

import json
import random
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser

# Test data
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "META", "NVDA", "AMD", "AMZN"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d"]
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d"]

class GoldenSignalsUser(FastHttpUser):
    """Simulates a typical user of the GoldenSignalsAI platform"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a user starts"""
        # Simulate initial page load
        self.client.get("/")

    @task(10)
    def get_signals(self):
        """Get trading signals - most common operation"""
        response = self.client.get("/api/v1/signals", name="/api/v1/signals")

        if response.status_code == 200:
            data = response.json()
            # Validate response structure
            assert "signals" in data, "Missing signals in response"

    @task(8)
    def get_market_data(self):
        """Get market data for a random symbol"""
        symbol = random.choice(SYMBOLS)
        response = self.client.get(
            f"/api/v1/market-data/{symbol}",
            name="/api/v1/market-data/[symbol]"
        )

        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data, "Missing symbol in response"
            assert "price" in data, "Missing price in response"

    @task(6)
    def get_historical_data(self):
        """Get historical data for analysis"""
        symbol = random.choice(SYMBOLS)
        period = random.choice(PERIODS)
        interval = random.choice(INTERVALS)

        response = self.client.get(
            f"/api/v1/market-data/{symbol}/historical",
            params={"period": period, "interval": interval},
            name="/api/v1/market-data/[symbol]/historical"
        )

        if response.status_code == 200:
            data = response.json()
            assert "data" in data, "Missing data in response"

    @task(5)
    def get_signal_insights(self):
        """Get detailed insights for a symbol"""
        symbol = random.choice(SYMBOLS)
        response = self.client.get(
            f"/api/v1/signals/{symbol}/insights",
            name="/api/v1/signals/[symbol]/insights"
        )

        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data, "Missing symbol in response"

    @task(4)
    def get_market_opportunities(self):
        """Check for market opportunities"""
        response = self.client.get("/api/v1/market/opportunities")

        if response.status_code == 200:
            data = response.json()
            assert "opportunities" in data, "Missing opportunities in response"

    @task(3)
    def get_performance_metrics(self):
        """Check system performance"""
        response = self.client.get("/api/v1/performance")

        if response.status_code == 200:
            data = response.json()
            assert "requests_per_endpoint" in data, "Missing metrics in response"

    @task(3)
    def get_monitoring_performance(self):
        """Get trading performance metrics"""
        response = self.client.get("/api/v1/monitoring/performance")

        if response.status_code == 200:
            data = response.json()
            assert "overall" in data, "Missing overall metrics"

    @task(2)
    def get_pipeline_stats(self):
        """Get signal pipeline statistics"""
        response = self.client.get("/api/v1/pipeline/stats")

        if response.status_code == 200:
            data = response.json()
            assert "filter_stats" in data, "Missing filter stats"

    @task(2)
    def get_signal_quality_report(self):
        """Get signal quality analysis"""
        response = self.client.get("/api/v1/signals/quality-report")

        if response.status_code == 200:
            data = response.json()
            assert "quality_metrics" in data, "Missing quality metrics"

    @task(1)
    def get_batch_signals(self):
        """Get signals for multiple symbols"""
        symbols = random.sample(SYMBOLS, k=3)

        response = self.client.post(
            "/api/v1/signals/batch",
            json={
                "symbols": symbols,
                "parameters": {
                    "min_confidence": 0.7,
                    "include_indicators": True
                }
            },
            name="/api/v1/signals/batch"
        )

        if response.status_code == 200:
            data = response.json()
            assert "signals" in data, "Missing signals in response"


class PowerUser(GoldenSignalsUser):
    """Simulates a power user with more aggressive usage patterns"""

    wait_time = between(0.5, 1.5)  # Shorter wait times

    @task(15)
    def rapid_signal_checking(self):
        """Power users check signals more frequently"""
        for _ in range(3):
            self.get_signals()
            self.client.get("/api/v1/monitoring/active-signals",
                          name="/api/v1/monitoring/active-signals")

    @task(5)
    def track_signal_entry(self):
        """Simulate entering a position"""
        response = self.client.post(
            "/api/v1/monitoring/track-entry",
            json={
                "signal_id": f"sig_{random.randint(100000, 999999)}",
                "symbol": random.choice(SYMBOLS),
                "entry_price": round(random.uniform(100, 500), 2),
                "quantity": random.randint(10, 100),
                "entry_time": "2024-12-23T10:30:00Z"
            },
            name="/api/v1/monitoring/track-entry"
        )

    @task(3)
    def websocket_simulation(self):
        """Simulate WebSocket-like rapid polling"""
        for _ in range(5):
            self.client.get("/api/v1/signals", name="/api/v1/signals [ws-sim]")
            self.client.get(
                f"/api/v1/market-data/{random.choice(SYMBOLS)}",
                name="/api/v1/market-data/[symbol] [ws-sim]"
            )


class MobileUser(GoldenSignalsUser):
    """Simulates mobile app usage patterns"""

    wait_time = between(2, 5)  # Longer wait times on mobile

    @task(20)
    def check_portfolio(self):
        """Mobile users frequently check their positions"""
        self.client.get("/api/v1/monitoring/active-signals",
                       name="/api/v1/monitoring/active-signals [mobile]")
        self.client.get("/api/v1/monitoring/performance",
                       name="/api/v1/monitoring/performance [mobile]")

    @task(10)
    def quick_signal_check(self):
        """Quick signal check on mobile"""
        self.client.get("/api/v1/signals", name="/api/v1/signals [mobile]")

    @task(5)
    def check_recommendations(self):
        """Check AI recommendations"""
        self.client.get("/api/v1/monitoring/recommendations",
                       name="/api/v1/monitoring/recommendations [mobile]")


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, **kwargs):
    """Custom request handler for additional metrics"""
    if response is not None:
        # Track slow requests
        if response_time > 1000:  # More than 1 second
            print(f"⚠️ Slow request: {name} took {response_time}ms")

        # Track errors
        if response.status_code >= 400:
            print(f"❌ Error {response.status_code} on {name}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("🚀 Starting load test...")
    print(f"Target host: {environment.host}")
    print(f"Total users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    print("\n📊 Load test completed!")

    # Print summary statistics
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")

    # Check if performance goals were met
    if stats.total.num_failures > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"\n⚠️ Failure rate: {failure_rate:.2f}%")

    if stats.total.avg_response_time > 200:
        print(f"\n⚠️ Average response time exceeds 200ms target")
    else:
        print(f"\n✅ Performance goals met!")
