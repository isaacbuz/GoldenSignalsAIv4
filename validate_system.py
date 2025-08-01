#!/usr/bin/env python3
"""
GoldenSignalsAI System Validation Script
Checks all components and configurations
"""

import os
import sys
import asyncio
import httpx
import redis
import asyncpg
from datetime import datetime
from typing import Dict, List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(name: str, status: bool, message: str = ""):
    """Print colored status message"""
    if status:
        print(f"{Colors.GREEN}✅ {name}{Colors.END}")
    else:
        print(f"{Colors.RED}❌ {name}{Colors.END}")
    if message:
        print(f"   {message}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

async def check_environment() -> Dict[str, bool]:
    """Check environment variables"""
    results = {}

    print("\n🔍 Checking Environment Variables...")

    # Check if .env exists
    env_exists = os.path.exists('.env')
    print_status(".env file exists", env_exists)
    results['.env'] = env_exists

    # Check critical environment variables
    critical_vars = [
        'DATABASE_URL',
        'REDIS_URL',
        'ALPHA_VANTAGE_API_KEY',
        'POLYGON_API_KEY',
        'FINNHUB_API_KEY'
    ]

    for var in critical_vars:
        value = os.getenv(var)
        exists = value is not None and value != ''
        if var.endswith('API_KEY'):
            # Don't show actual API keys
            print_status(f"{var}", exists,
                        "Set" if exists else "Not set - will use mock data")
        else:
            print_status(f"{var}", exists,
                        f"Value: {value[:20]}..." if exists else "Not set")
        results[var] = exists

    return results

async def check_services() -> Dict[str, bool]:
    """Check if services are running"""
    results = {}

    print("\n🔍 Checking Services...")

    # Check Redis
    try:
        r = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        r.ping()
        print_status("Redis", True, "Connected successfully")
        results['redis'] = True
    except Exception as e:
        print_status("Redis", False, f"Error: {str(e)}")
        results['redis'] = False

    # Check PostgreSQL
    try:
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            user=os.getenv('DB_USER', 'goldensignals'),
            password=os.getenv('DB_PASSWORD', 'password'),
            database=os.getenv('DB_NAME', 'goldensignals')
        )
        await conn.close()
        print_status("PostgreSQL", True, "Connected successfully")
        results['postgresql'] = True
    except Exception as e:
        print_status("PostgreSQL", False, f"Error: {str(e)}")
        print_warning("Will use SQLite as fallback")
        results['postgresql'] = False

    # Check Backend API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000", timeout=5.0)
            if response.status_code == 200:
                print_status("Backend API", True, "Running on http://localhost:8000")
                results['backend'] = True
            else:
                print_status("Backend API", False, f"Status code: {response.status_code}")
                results['backend'] = False
    except Exception as e:
        print_status("Backend API", False, "Not running")
        results['backend'] = False

    # Check Frontend
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000", timeout=5.0)
            if response.status_code == 200:
                print_status("Frontend", True, "Running on http://localhost:3000")
                results['frontend'] = True
            else:
                print_status("Frontend", False, f"Status code: {response.status_code}")
                results['frontend'] = False
    except Exception as e:
        print_status("Frontend", False, "Not running")
        results['frontend'] = False

    return results

async def check_api_endpoints() -> Dict[str, bool]:
    """Check API endpoints"""
    results = {}

    print("\n🔍 Checking API Endpoints...")

    endpoints = [
        ("/", "Root"),
        ("/api/v1/market-data/SPY", "Market Data"),
        ("/api/v1/signals", "Signals"),
        ("/api/v1/market/opportunities", "Market Opportunities"),
        ("/docs", "API Documentation")
    ]

    async with httpx.AsyncClient() as client:
        for endpoint, name in endpoints:
            try:
                response = await client.get(f"http://localhost:8000{endpoint}", timeout=5.0)
                success = response.status_code == 200
                print_status(f"{name} ({endpoint})", success,
                           f"Status: {response.status_code}")
                results[endpoint] = success
            except Exception as e:
                print_status(f"{name} ({endpoint})", False, "Failed to connect")
                results[endpoint] = False

    return results

async def check_websocket() -> bool:
    """Check WebSocket connection"""
    print("\n🔍 Checking WebSocket...")

    try:
        import websockets
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print_status("WebSocket", True, "Connected successfully")
            return True
    except Exception as e:
        print_status("WebSocket", False, f"Error: {str(e)}")
        return False

def check_python_packages() -> Dict[str, bool]:
    """Check required Python packages"""
    results = {}

    print("\n🔍 Checking Python Packages...")

    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'yfinance',
        'redis',
        'asyncpg',
        'sqlalchemy',
        'websockets',
        'httpx',
        'pytz'
    ]

    for package in required_packages:
        try:
            __import__(package)
            print_status(f"{package}", True)
            results[package] = True
        except ImportError:
            print_status(f"{package}", False, "Not installed")
            results[package] = False

    return results

def generate_report(env_results: Dict, service_results: Dict,
                   api_results: Dict, ws_result: bool,
                   package_results: Dict):
    """Generate final validation report"""
    print("\n" + "="*50)
    print("📊 VALIDATION REPORT")
    print("="*50)

    # Calculate scores
    env_score = sum(1 for v in env_results.values() if v)
    service_score = sum(1 for v in service_results.values() if v)
    api_score = sum(1 for v in api_results.values() if v)
    package_score = sum(1 for v in package_results.values() if v)

    total_checks = (len(env_results) + len(service_results) +
                   len(api_results) + 1 + len(package_results))
    passed_checks = env_score + service_score + api_score + (1 if ws_result else 0) + package_score

    print(f"\n📈 Overall Score: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)")

    print(f"\n🔧 Environment: {env_score}/{len(env_results)}")
    print(f"🖥️  Services: {service_score}/{len(service_results)}")
    print(f"🌐 API Endpoints: {api_score}/{len(api_results)}")
    print(f"🔌 WebSocket: {'✅' if ws_result else '❌'}")
    print(f"📦 Python Packages: {package_score}/{len(package_results)}")

    # Recommendations
    print("\n💡 Recommendations:")

    if not env_results.get('ALPHA_VANTAGE_API_KEY'):
        print_warning("Add API keys to .env for live market data")

    if not service_results.get('postgresql'):
        print_warning("PostgreSQL not running - using SQLite fallback")

    if not service_results.get('backend'):
        print_warning("Start backend with: python simple_backend.py")

    if not service_results.get('frontend'):
        print_warning("Start frontend with: cd frontend && npm run dev")

    missing_packages = [p for p, v in package_results.items() if not v]
    if missing_packages:
        print_warning(f"Install missing packages: pip install {' '.join(missing_packages)}")

    # Final status
    print("\n" + "="*50)
    if passed_checks == total_checks:
        print(f"{Colors.GREEN}🎉 System is fully operational!{Colors.END}")
    elif passed_checks >= total_checks * 0.8:
        print(f"{Colors.YELLOW}⚠️  System is mostly operational with some warnings{Colors.END}")
    else:
        print(f"{Colors.RED}❌ System needs configuration{Colors.END}")
    print("="*50)

async def main():
    """Run all validation checks"""
    print("🚀 GoldenSignalsAI System Validation")
    print("="*50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run checks
    env_results = await check_environment()
    package_results = check_python_packages()
    service_results = await check_services()

    # Only check API and WebSocket if backend is running
    if service_results.get('backend'):
        api_results = await check_api_endpoints()
        ws_result = await check_websocket()
    else:
        api_results = {}
        ws_result = False
        print_info("Skipping API and WebSocket checks (backend not running)")

    # Generate report
    generate_report(env_results, service_results, api_results, ws_result, package_results)

    print("\n✨ To start the system, run: ./start_all.sh")
    print("📚 For more information, see: PERFECT_IMPLEMENTATION_SUMMARY.md")

if __name__ == "__main__":
    asyncio.run(main())
