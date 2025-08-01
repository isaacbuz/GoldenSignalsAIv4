#!/usr/bin/env python3
"""
Test script to demonstrate after-hours data handling
Shows how the system handles data requests when markets are closed
"""

import requests
import json
from datetime import datetime
import time
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print info message"""
    print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")

def test_market_status():
    """Test market status endpoint"""
    print_header("Testing Market Status")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/market-data/status")
        data = response.json()

        if data["is_open"]:
            print_success(f"Market is OPEN")
        else:
            print_warning(f"Market is CLOSED")

        print_info(f"Current time: {data['current_time']}")
        print_info(f"Market hours: {data['market_hours']['open']} - {data['market_hours']['close']} {data['timezone']}")
        print_info(f"Next open: {data['next_open']}")

        return data["is_open"]

    except Exception as e:
        print_error(f"Failed to get market status: {e}")
        return None

def test_market_data(symbol="AAPL"):
    """Test market data endpoint with after-hours handling"""
    print_header(f"Testing Market Data for {symbol}")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/market-data/{symbol}")

        if response.status_code == 200:
            data = response.json()

            print_success(f"Got market data for {symbol}")
            print_info(f"Price: ${data['price']:.2f}")
            print_info(f"Change: {data['change']:.2f} ({data['change_percent']:.2f}%)")
            print_info(f"Volume: {data['volume']:,}")
            print_info(f"Data source: {data.get('data_source', 'unknown')}")

            if "warning" in data:
                print_warning(f"Warning: {data['warning']}")

        elif response.status_code == 503 or response.status_code == 500:
            try:
                error_data = response.json()
                detail = error_data.get("detail", {})

                # Handle both dict and string detail formats
                if isinstance(detail, dict):
                    print_warning(f"Service unavailable: {detail.get('message', 'Unknown error')}")
                    print_info(f"Reason: {detail.get('reason', 'unknown')}")
                    print_info(f"Suggested action: {detail.get('suggested_action', 'N/A')}")
                    print_info(f"Is recoverable: {detail.get('is_recoverable', False)}")
                else:
                    print_warning(f"Service unavailable: {detail}")
            except:
                print_error(f"Failed with status code: {response.status_code}")
                print_error(f"Response: {response.text}")

        else:
            print_error(f"Failed with status code: {response.status_code}")
            print_error(f"Response: {response.text}")

    except Exception as e:
        print_error(f"Failed to get market data: {e}")

def test_signal_generation(symbol="AAPL"):
    """Test signal generation with after-hours handling"""
    print_header(f"Testing Signal Generation for {symbol}")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/signals/{symbol}")

        if response.status_code == 200:
            data = response.json()

            print_success(f"Got signal for {symbol}")
            print_info(f"Signal: {data['signal']}")
            print_info(f"Confidence: {data['confidence']:.2%}")
            print_info(f"Price Target: ${data['price_target']:.2f}")
            print_info(f"Stop Loss: ${data['stop_loss']:.2f}")
            print_info(f"Risk Score: {data['risk_score']:.2f}")

            if data.get("is_after_hours"):
                print_warning("Signal generated using after-hours/cached data")

            market_status = data.get("market_status", {})
            if not market_status.get("is_open"):
                print_warning(f"Market Status: {market_status.get('reason', 'Unknown')}")
                if market_status.get("next_open"):
                    print_info(f"Next market open: {market_status['next_open']}")

            # Display some indicators
            indicators = data.get("indicators", {})
            if indicators:
                print_info("\nKey Indicators:")
                for key in ["rsi", "macd", "bb_width", "volume_ratio"]:
                    if key in indicators:
                        print_info(f"  {key.upper()}: {indicators[key]:.2f}")

                if indicators.get("after_hours_data"):
                    print_warning("  Using after-hours data")

        else:
            print_error(f"Failed with status code: {response.status_code}")
            print_error(f"Response: {response.text}")

    except Exception as e:
        print_error(f"Failed to get signal: {e}")

def test_multiple_symbols():
    """Test multiple symbols to see cache behavior"""
    print_header("Testing Multiple Symbols")

    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "INVALID_SYMBOL"]

    for symbol in symbols:
        print(f"\n{Fore.CYAN}--- Testing {symbol} ---{Style.RESET_ALL}")

        # Test market data
        try:
            response = requests.get(f"{BASE_URL}/api/v1/market-data/{symbol}")

            if response.status_code == 200:
                data = response.json()
                print_success(f"{symbol}: ${data['price']:.2f} ({data.get('data_source', 'unknown')})")
                if "warning" in data:
                    print_warning(f"{symbol}: {data['warning']}")
            else:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", {})

                    # Handle both dict and string detail formats
                    if isinstance(detail, dict):
                        print_warning(f"{symbol}: {detail.get('reason', 'error')} - {detail.get('message', 'Unknown error')}")
                    else:
                        print_warning(f"{symbol}: {detail}")
                except:
                    print_error(f"{symbol}: Failed with status {response.status_code}")

        except Exception as e:
            print_error(f"{symbol}: Failed - {e}")

        time.sleep(0.5)  # Small delay between requests

def main():
    """Main test function"""
    print(f"{Fore.MAGENTA}🚀 After-Hours Data Handling Test{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Testing GoldenSignalsAI V3 Market Data Service{Style.RESET_ALL}")

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print_error("Server is not responding properly")
            return
    except:
        print_error("Cannot connect to server. Make sure it's running on port 8000")
        return

    # Run tests
    is_market_open = test_market_status()

    if is_market_open is False:
        print_warning("\nMarket is closed - perfect for testing after-hours handling!")
    elif is_market_open:
        print_info("\nMarket is open - live data should be available")

    # Test individual symbol
    test_market_data("AAPL")
    test_signal_generation("AAPL")

    # Test multiple symbols
    test_multiple_symbols()

    print_header("Test Complete")

    if is_market_open is False:
        print_info("During market hours, the system will fetch live data")
        print_info("After hours, it intelligently uses cached data when available")
        print_info("This ensures 24/7 availability for analysis and backtesting")
    else:
        print_info("Live market data is being fetched")
        print_info("Data is being cached for after-hours availability")

if __name__ == "__main__":
    main()
