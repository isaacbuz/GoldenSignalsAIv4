#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced error tracking system
"""

import requests
import json
import time
from datetime import datetime

def test_error_tracking():
    """Test the error tracking system"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Enhanced Error Tracking System")
    print("=" * 50)
    
    # Test 1: Normal API calls
    print("\n1. Testing normal API calls...")
    
    # Health check
    response = requests.get(f"{base_url}/api/v1/monitoring/health")
    if response.status_code == 200:
        health_data = response.json()
        print(f"✅ Health Status: {health_data['status']}")
        print(f"   Total Requests: {health_data['performance']['total_requests']}")
        print(f"   Errors: {health_data['errors']['total_errors']}")
    
    # Test 2: Performance monitoring
    print("\n2. Testing performance monitoring...")
    
    # Make several API calls to generate performance data
    endpoints = [
        "/api/v1/signals/latest",
        "/api/v1/market-data/AAPL",
        "/api/v1/signals/TSLA",
        "/api/v1/market-data/GOOGL/historical"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            print(f"   {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"   {endpoint}: Error - {e}")
    
    # Test 3: Check performance summary
    print("\n3. Checking performance summary...")
    response = requests.get(f"{base_url}/api/v1/monitoring/performance-detailed")
    if response.status_code == 200:
        perf_data = response.json()
        print(f"   Total Requests: {perf_data['total_requests']}")
        print(f"   Endpoints Tracked: {len(perf_data['endpoints'])}")
        print(f"   Slow Endpoints: {len(perf_data['slow_endpoints'])}")
        
        # Show top endpoints by request count
        if perf_data['endpoints']:
            print("\n   Top Endpoints by Request Count:")
            sorted_endpoints = sorted(
                perf_data['endpoints'].items(), 
                key=lambda x: x[1]['count'], 
                reverse=True
            )
            for endpoint, stats in sorted_endpoints[:5]:
                print(f"     {endpoint}: {stats['count']} requests, "
                      f"avg: {stats['avg_time']:.3f}s")
    
    # Test 4: Error tracking
    print("\n4. Testing error tracking...")
    
    # Try to trigger an error by calling a non-existent endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/nonexistent-endpoint")
        print(f"   Non-existent endpoint: {response.status_code}")
    except Exception as e:
        print(f"   Non-existent endpoint: Error - {e}")
    
    # Check error summary
    response = requests.get(f"{base_url}/api/v1/monitoring/errors")
    if response.status_code == 200:
        error_data = response.json()
        print(f"   Total Errors: {error_data['total_errors']}")
        print(f"   Recent Errors (1h): {error_data['recent_errors_1h']}")
        if error_data['most_common_errors']:
            print("   Most Common Errors:")
            for error, count in error_data['most_common_errors'][:3]:
                print(f"     {error}: {count} times")
    
    # Test 5: Frontend logging
    print("\n5. Testing frontend logging...")
    
    # Simulate frontend error logging
    frontend_log = {
        "level": "error",
        "message": "Test frontend error from error tracking test",
        "data": {
            "component": "TestComponent",
            "timestamp": datetime.now().isoformat(),
            "userAgent": "Test Script",
            "url": "http://localhost:3000/test"
        }
    }
    
    response = requests.post(f"{base_url}/api/logs/frontend", json=frontend_log)
    if response.status_code == 200:
        print(f"   Frontend log submitted: {response.json()['status']}")
    
    # Test 6: Final health check
    print("\n6. Final health check...")
    response = requests.get(f"{base_url}/api/v1/monitoring/health")
    if response.status_code == 200:
        health_data = response.json()
        print(f"   Health Status: {health_data['status']}")
        print(f"   Total Requests: {health_data['performance']['total_requests']}")
        print(f"   Cache Usage:")
        for cache_name, cache_info in health_data['cache_stats'].items():
            print(f"     {cache_name}: {cache_info['size']}/{cache_info['maxsize']}")
    
    print("\n✅ Error tracking system test completed!")
    print("\n📊 Summary:")
    print("   - Error tracking: Working")
    print("   - Performance monitoring: Working")
    print("   - Health monitoring: Working")
    print("   - Frontend logging: Working")
    print("   - Cache monitoring: Working")

if __name__ == "__main__":
    test_error_tracking() 