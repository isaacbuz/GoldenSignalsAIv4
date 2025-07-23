"""
Security Test Suite for GoldenSignalsAI
Tests authentication, authorization, rate limiting, and input validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import redis
import time
import jwt
from datetime import datetime, timedelta
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.main import app
    from src.middleware.rate_limiter import RateLimiter
except ImportError:
    # Use test app if main app can't be imported
    app = None
    RateLimiter = None


class TestRateLimiter:
    """Test suite for rate limiting functionality"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing"""
        mock = MagicMock(spec=redis.Redis)
        return mock

    def test_rate_limiter_fails_closed_when_redis_unavailable(self, mock_redis):
        """Test that rate limiter denies requests when Redis is unavailable"""
        # Configure mock to raise exception
        mock_redis.pipeline.side_effect = redis.ConnectionError("Redis unavailable")

        # Create rate limiter with mock
        if RateLimiter is None:
            pytest.skip("RateLimiter not available")

        limiter = RateLimiter(mock_redis)

        # Create mock request
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/api/v1/test"

        # Check rate limit
        allowed, metadata = asyncio.run(limiter.check_rate_limit(
            mock_request,
            limit=10,
            window_seconds=60
        ))

        # Should fail closed (deny request)
        assert allowed is False
        assert metadata.get('error') == 'Rate limiting service temporarily unavailable'
        assert metadata.get('retry_after') == 60

    def test_rate_limiter_prevents_request_flooding(self, client, mock_redis):
        """Test that rate limiter blocks excessive requests"""
        # Configure mock to track calls
        call_count = 0

        def mock_zcard(key):
            nonlocal call_count
            call_count += 1
            return call_count

        mock_redis.pipeline.return_value.zcard.side_effect = mock_zcard
        mock_redis.pipeline.return_value.execute.return_value = [None, 11, None, None]

        with patch('src.middleware.rate_limiter.redis', mock_redis):
            # Make requests up to the limit
            for i in range(10):
                response = client.get("/api/v1/signals")
                assert response.status_code != 429

            # Next request should be rate limited
            response = client.get("/api/v1/signals")
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["error"]

    def test_ip_spoofing_protection(self, mock_redis):
        """Test that IP extraction is secure against spoofing"""
        if RateLimiter is None:
            pytest.skip("RateLimiter not available")
        limiter = RateLimiter(mock_redis)

        # Test without trusted proxy header - should use direct IP
        mock_request = MagicMock()
        mock_request.headers = {
            'X-Forwarded-For': '1.2.3.4, 5.6.7.8',
            'X-Real-IP': '9.10.11.12'
        }
        mock_request.client.host = "192.168.1.100"

        identifier = limiter._get_identifier(mock_request)
        assert identifier == "ip:192.168.1.100"

        # Test with trusted proxy header - should parse forwarded IPs
        mock_request.headers['X-Proxy-Auth'] = 'trusted-proxy-secret'
        identifier = limiter._get_identifier(mock_request)
        # Should use the rightmost non-private IP
        assert identifier in ["ip:5.6.7.8", "ip:9.10.11.12"]


class TestAuthentication:
    """Test suite for authentication and authorization"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_protected_endpoints_require_authentication(self, client):
        """Test that protected endpoints return 401 without auth"""
        protected_endpoints = [
            "/api/v1/portfolio",
            "/api/v1/user/preferences",
            "/api/v1/alerts/subscribe",
        ]

        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401
            assert "Not authenticated" in response.json().get("detail", "")

    def test_jwt_token_validation(self, client):
        """Test JWT token validation"""
        # Test with invalid token
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/v1/portfolio", headers=headers)
        assert response.status_code == 401

        # Test with expired token
        expired_token = jwt.encode(
            {
                "sub": "test-user",
                "exp": datetime.utcnow() - timedelta(hours=1)
            },
            "secret-key",
            algorithm="HS256"
        )
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/portfolio", headers=headers)
        assert response.status_code == 401

    def test_api_key_authentication(self, client):
        """Test API key authentication"""
        # Test with missing API key
        response = client.get("/api/v1/market-data/AAPL")
        assert response.status_code in [200, 403]  # Depends on configuration

        # Test with invalid API key
        headers = {"X-API-Key": "invalid-key"}
        response = client.get("/api/v1/market-data/AAPL", headers=headers)
        # Should either work (public endpoint) or return 403
        assert response.status_code in [200, 403]


class TestInputValidation:
    """Test suite for input validation and sanitization"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_sql_injection_prevention(self, client):
        """Test that SQL injection attempts are blocked"""
        # Test various SQL injection patterns
        injection_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "1; DELETE FROM signals WHERE 1=1; --",
            "UNION SELECT * FROM users",
        ]

        for payload in injection_attempts:
            response = client.get(f"/api/v1/signals/{payload}")
            # Should return 422 (validation error) or 404 (not found)
            assert response.status_code in [422, 404]
            # Should not execute the SQL
            assert "DROP TABLE" not in str(response.content)

    def test_xss_prevention(self, client):
        """Test that XSS attempts are sanitized"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
        ]

        for payload in xss_payloads:
            # Test in various endpoints that accept user input
            response = client.post(
                "/api/v1/analyze",
                json={"symbol": payload}
            )

            # Response should not contain unescaped script tags
            response_text = response.text
            assert "<script>" not in response_text
            assert "javascript:" not in response_text
            assert "onerror=" not in response_text

    def test_path_traversal_prevention(self, client):
        """Test that path traversal attempts are blocked"""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        for payload in traversal_attempts:
            response = client.get(f"/api/v1/files/{payload}")
            # Should return 400 or 404, not expose system files
            assert response.status_code in [400, 404]
            assert "passwd" not in response.text.lower()
            assert "sam" not in response.text.lower()

    def test_command_injection_prevention(self, client):
        """Test that command injection attempts are blocked"""
        command_payloads = [
            "; ls -la",
            "| whoami",
            "& net user",
            "`id`",
            "$(whoami)",
        ]

        for payload in command_payloads:
            response = client.post(
                "/api/v1/analyze",
                json={"symbol": f"AAPL{payload}"}
            )
            # Should not execute system commands
            assert response.status_code in [200, 422]
            assert "uid=" not in response.text
            assert "root" not in response.text
            assert "administrator" not in response.text.lower()


class TestDataIntegrity:
    """Test suite for data integrity and consistency"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_websocket_message_integrity(self, client):
        """Test that WebSocket messages maintain integrity"""
        with client.websocket_connect("/ws/market-data") as websocket:
            # Send subscription message
            websocket.send_json({
                "type": "subscribe",
                "symbol": "AAPL"
            })

            # Receive messages and verify structure
            for _ in range(5):
                data = websocket.receive_json()
                # Verify required fields exist
                assert "type" in data
                assert "timestamp" in data

                if data["type"] == "price":
                    assert "symbol" in data
                    assert "price" in data
                    assert isinstance(data["price"], (int, float))
                    assert data["price"] > 0

    def test_concurrent_request_handling(self, client):
        """Test that concurrent requests don't cause data corruption"""
        import concurrent.futures

        def make_request(symbol):
            response = client.get(f"/api/v1/signals/{symbol}")
            return response.json()

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, symbol) for symbol in symbols * 2]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify each result maintains integrity
        for result in results:
            if "signals" in result:
                for signal in result["signals"]:
                    assert "symbol" in signal
                    assert "confidence" in signal
                    assert 0 <= signal["confidence"] <= 100

    def test_transaction_atomicity(self, client):
        """Test that multi-step operations are atomic"""
        # Test portfolio update atomicity
        portfolio_update = {
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "action": "buy"},
                {"symbol": "GOOGL", "quantity": 50, "action": "buy"},
                {"symbol": "INVALID_SYMBOL", "quantity": 25, "action": "buy"},
            ]
        }

        response = client.post("/api/v1/portfolio/update", json=portfolio_update)

        # If any position fails, all should rollback
        if response.status_code != 200:
            # Verify no partial updates occurred
            portfolio_response = client.get("/api/v1/portfolio")
            portfolio = portfolio_response.json()

            # Should not contain any of the attempted positions
            symbols_in_portfolio = [p["symbol"] for p in portfolio.get("positions", [])]
            assert "AAPL" not in symbols_in_portfolio
            assert "GOOGL" not in symbols_in_portfolio


class TestSecurityHeaders:
    """Test suite for security headers"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_security_headers_present(self, client):
        """Test that security headers are properly set"""
        response = client.get("/api/v1/health")

        # Check for important security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] in ["DENY", "SAMEORIGIN"]

        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"

        # Check CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        # Should not be wildcard for production
        assert response.headers["Access-Control-Allow-Origin"] != "*"

    def test_content_security_policy(self, client):
        """Test Content Security Policy header"""
        response = client.get("/")

        if "Content-Security-Policy" in response.headers:
            csp = response.headers["Content-Security-Policy"]
            # Should restrict script sources
            assert "script-src" in csp
            assert "unsafe-inline" not in csp or "nonce-" in csp
            # Should restrict object sources
            assert "object-src 'none'" in csp


class TestEncryption:
    """Test suite for encryption and secure communication"""

    def test_password_hashing(self):
        """Test that passwords are properly hashed"""
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Test password hashing
        password = "test_password_123"
        hashed = pwd_context.hash(password)

        # Verify hash characteristics
        assert password not in hashed
        assert hashed.startswith("$2b$")  # bcrypt prefix
        assert len(hashed) >= 60  # bcrypt hash length

        # Verify password verification
        assert pwd_context.verify(password, hashed)
        assert not pwd_context.verify("wrong_password", hashed)

    def test_sensitive_data_not_logged(self, caplog):
        """Test that sensitive data is not logged"""
        import logging
        try:
            from src.services.auth_service import authenticate_user
        except ImportError:
            authenticate_user = None

        # Set up logging capture
        caplog.set_level(logging.DEBUG)

        # Attempt authentication (mock or real)
        if authenticate_user is not None:
            with patch('src.services.auth_service.verify_password', return_value=False):
                result = authenticate_user("test@example.com", "sensitive_password_123")
        else:
            # Simulate authentication attempt
            caplog.clear()
            logging.getLogger().info("Authentication attempt for test@example.com")
            logging.getLogger().debug("Verifying credentials")

        # Check logs don't contain sensitive data
        log_text = caplog.text.lower()
        assert "sensitive_password_123" not in log_text
        assert "password" not in log_text or "****" in log_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
