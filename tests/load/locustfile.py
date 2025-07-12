"""
Load testing for GoldenSignalsAI API.

Run with: locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random
import json

class GoldenSignalsUser(HttpUser):
    """Simulated user for load testing."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get auth token."""
        response = self.client.post("/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def get_signals(self):
        """Get signals for random symbol."""
        symbol = random.choice(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
        self.client.get(f"/api/v1/signals/{symbol}")
    
    @task(2)
    def get_latest_signals(self):
        """Get latest signals."""
        self.client.get("/api/v1/signals/latest")
    
    @task(1)
    def generate_signals(self):
        """Generate new signals."""
        symbols = random.sample(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], 3)
        self.client.get(f"/api/v1/signals/generate", params={"symbols": symbols})
    
    @task(2)
    def get_portfolio(self):
        """Get portfolio status."""
        self.client.get("/api/v1/portfolio/status")
    
    @task(1)
    def health_check(self):
        """Check system health."""
        self.client.get("/api/v1/health/")

class AdminUser(HttpUser):
    """Simulated admin user."""
    
    wait_time = between(5, 10)
    
    def on_start(self):
        """Login as admin."""
        response = self.client.post("/auth/login", json={
            "username": "admin",
            "password": "adminpass"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task
    def get_system_metrics(self):
        """Get system metrics."""
        self.client.get("/api/v1/admin/metrics")
    
    @task
    def get_agent_status(self):
        """Get agent status."""
        self.client.get("/api/v1/agents/status")
