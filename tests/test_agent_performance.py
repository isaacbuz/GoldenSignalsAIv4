import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_agent_performance_summary(monkeypatch):
    class DummyTracker:
        def get_summary(self, agent_id):
            return {"agent_id": agent_id, "total_signals": 10, "wins": 7, "losses": 3, "win_rate": 70.0, "last_signal_time": 1234567890}
    monkeypatch.setattr("application.monitoring.agent_performance_tracker.AgentPerformanceTracker", DummyTracker)
    res = client.post("/api/v1/agent_performance/summary", json={"agent_id": "agent-123"})
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert data["summary"]["win_rate"] == 70.0
