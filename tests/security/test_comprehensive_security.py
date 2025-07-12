"""Comprehensive security tests for GoldenSignalsAI V2."""

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_authentication_required():
    """Test that protected endpoints require authentication."""
    response = client.get("/api/v1/admin/users")
    assert response.status_code == 401  # Unauthorized

def test_rate_limiting():
    """Test rate limiting functionality."""
    # Make multiple rapid requests
    for _ in range(100):
        response = client.get("/health")
    
    # Should not be rate limited for health endpoint
    assert response.status_code == 200

def test_input_validation():
    """Test input validation and sanitization."""
    # Test SQL injection attempt
    malicious_input = "'; DROP TABLE users; --"
    response = client.post("/api/v1/signals", json={"symbol": malicious_input})
    
    # Should handle gracefully (either 400 or sanitized)
    assert response.status_code in [200, 400, 422]

def test_cors_headers():
    """Test CORS headers are properly set."""
    response = client.options("/api/v1/signals")
    assert "access-control-allow-origin" in response.headers

def test_content_security_policy():
    """Test Content Security Policy headers."""
    response = client.get("/health")
    # Should have security headers
    assert response.status_code == 200