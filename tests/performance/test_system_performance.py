"""
Performance tests for the GoldenSignalsAI V2 trading system.
"""
import pytest
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics


class TestPerformance:
    """Performance tests for the trading system"""
    
    base_url = "http://localhost:8000"
    
    @pytest.fixture(autouse=True)
    def check_backend(self):
        """Check if backend is running before tests"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            if response.status_code != 200:
                pytest.skip("Backend not running")
        except:
            pytest.skip("Backend not running")
    
    def test_api_response_time(self):
        """Test API endpoint response times"""
        endpoints = [
            "/api/v1/signals",
            "/api/v1/market-data/SPY",
            "/api/v1/signals/SPY/insights",
            "/api/v1/market/opportunities"
        ]
        
        results = {}
        
        for endpoint in endpoints:
            response_times = []
            
            # Make 20 requests to each endpoint
            for _ in range(20):
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}")
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if response_times:
                results[endpoint] = {
                    'avg': statistics.mean(response_times),
                    'median': statistics.median(response_times),
                    'p95': np.percentile(response_times, 95),
                    'p99': np.percentile(response_times, 99)
                }
        
        # Assert performance requirements
        for endpoint, metrics in results.items():
            assert metrics['avg'] < 200, f"{endpoint} average response time > 200ms"
            assert metrics['p95'] < 500, f"{endpoint} p95 response time > 500ms"
            assert metrics['p99'] < 1000, f"{endpoint} p99 response time > 1000ms"
    
    @pytest.mark.slow
    def test_concurrent_load(self):
        """Test system performance under concurrent load"""
        def make_request(request_id):
            """Make a request and return timing info"""
            endpoint = f"/api/v1/signals"
            start_time = time.time()
            
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                end_time = time.time()
                
                return {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'response_time': (end_time - start_time) * 1000,
                    'success': response.status_code == 200
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'request_id': request_id,
                    'status_code': 0,
                    'response_time': (end_time - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
        
        # Test with increasing concurrent requests
        for num_concurrent in [10, 20, 50]:
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                # Submit all requests
                futures = []
                for i in range(num_concurrent):
                    future = executor.submit(make_request, i)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Analyze results
            successful_requests = [r for r in results if r['success']]
            response_times = [r['response_time'] for r in successful_requests]
            
            success_rate = len(successful_requests) / len(results)
            
            # Assert performance under load
            assert success_rate >= 0.95, f"Success rate < 95% with {num_concurrent} concurrent requests"
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                p95_response_time = np.percentile(response_times, 95)
                
                assert avg_response_time < 1000, f"Average response time > 1s with {num_concurrent} concurrent requests"
                assert p95_response_time < 2000, f"P95 response time > 2s with {num_concurrent} concurrent requests"
    
    def test_historical_data_performance(self):
        """Test performance of historical data retrieval"""
        periods = ["1d", "5d", "1mo"]
        intervals = ["5m", "1h", "1d"]
        
        results = []
        
        for period in periods:
            for interval in intervals:
                start_time = time.time()
                response = requests.get(
                    f"{self.base_url}/api/v1/market-data/SPY/historical",
                    params={"period": period, "interval": interval}
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        'period': period,
                        'interval': interval,
                        'response_time': (end_time - start_time) * 1000,
                        'data_points': len(data.get('data', []))
                    })
        
        # Assert reasonable response times for different data sizes
        for result in results:
            # Longer periods should still respond quickly due to caching
            assert result['response_time'] < 3000, f"Historical data request too slow: {result}"
    
    def test_cache_effectiveness(self):
        """Test that caching improves performance"""
        endpoint = "/api/v1/signals/SPY"
        
        # First request (cache miss)
        start_time = time.time()
        response1 = requests.get(f"{self.base_url}{endpoint}")
        first_request_time = (time.time() - start_time) * 1000
        
        assert response1.status_code == 200
        
        # Immediate second request (should hit cache)
        start_time = time.time()
        response2 = requests.get(f"{self.base_url}{endpoint}")
        second_request_time = (time.time() - start_time) * 1000
        
        assert response2.status_code == 200
        
        # Cache should make second request significantly faster
        assert second_request_time < first_request_time * 0.5, "Cache not effective"
    
    @pytest.mark.slow
    def test_sustained_load(self):
        """Test system performance under sustained load"""
        duration_seconds = 30
        requests_per_second = 10
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_times = []
        errors = []
        
        request_count = 0
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # Make requests for this second
            for _ in range(requests_per_second):
                req_start = time.time()
                try:
                    response = requests.get(f"{self.base_url}/api/v1/signals", timeout=2)
                    req_end = time.time()
                    
                    request_times.append((req_end - req_start) * 1000)
                    request_count += 1
                    
                    if response.status_code != 200:
                        errors.append(f"Status code: {response.status_code}")
                except Exception as e:
                    errors.append(str(e))
                    request_count += 1
            
            # Sleep to maintain rate
            batch_duration = time.time() - batch_start
            if batch_duration < 1.0:
                time.sleep(1.0 - batch_duration)
        
        # Calculate metrics
        error_rate = len(errors) / request_count if request_count > 0 else 1.0
        
        if request_times:
            avg_response_time = statistics.mean(request_times)
            p95_response_time = np.percentile(request_times, 95)
            
            # Assert sustained performance
            assert error_rate < 0.05, f"Error rate too high: {error_rate * 100:.1f}%"
            assert avg_response_time < 500, f"Average response time under load: {avg_response_time:.1f}ms"
            assert p95_response_time < 1000, f"P95 response time under load: {p95_response_time:.1f}ms" 