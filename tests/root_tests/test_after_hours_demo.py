#!/usr/bin/env python3
"""
Demo script for after-hours data handling using mock data
Shows how the system handles data requests when markets are closed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'services'))

from market_data_service_mock import MockMarketDataService
from colorama import init, Fore, Style
from datetime import datetime
import pandas as pd

# Initialize colorama for colored output
init(autoreset=True)

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

def demo_market_hours():
    """Demonstrate market hours checking"""
    print_header("Market Hours Detection")
    
    service = MockMarketDataService()
    market_hours = service.check_market_hours()
    
    if market_hours.is_open:
        print_success("Market is OPEN")
    else:
        print_warning("Market is CLOSED")
    
    print_info(f"Current time: {market_hours.current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print_info(f"Market hours: {market_hours.market_open.strftime('%H:%M')} - {market_hours.market_close.strftime('%H:%M')} {market_hours.timezone}")
    print_info(f"Status: {market_hours.reason}")
    
    if not market_hours.is_open:
        print_info(f"Next market open: {market_hours.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    return market_hours

def demo_data_fetching(service, symbol="AAPL"):
    """Demonstrate data fetching with error handling"""
    print_header(f"Data Fetching for {symbol}")
    
    # Fetch real-time data
    tick, error = service.fetch_real_time_data(symbol)
    
    if tick:
        print_success(f"Got market data for {symbol}")
        print_info(f"Price: ${tick.price:.2f}")
        print_info(f"Change: ${tick.change:.2f} ({tick.change_percent:.2f}%)")
        print_info(f"Volume: {tick.volume:,}")
        print_info(f"Bid/Ask: ${tick.bid:.2f} / ${tick.ask:.2f}")
        print_info(f"Spread: ${tick.spread:.4f}")
        print_info(f"Timestamp: {tick.timestamp}")
    elif error:
        print_warning(f"Data unavailable: {error.reason.value}")
        print_info(f"Message: {error.message}")
        print_info(f"Is recoverable: {error.is_recoverable}")
        print_info(f"Suggested action: {error.suggested_action}")

def demo_historical_data(service, symbol="AAPL"):
    """Demonstrate historical data handling"""
    print_header(f"Historical Data for {symbol}")
    
    data, error = service.get_historical_data(symbol, period="3mo")
    
    if not data.empty:
        print_success(f"Got {len(data)} days of historical data")
        
        # Show recent data
        recent = data.tail(5)
        print_info("\nRecent 5 days:")
        print(recent[['Close', 'Volume', 'RSI', 'MACD']].round(2))
        
        # Show current indicators
        latest = data.iloc[-1]
        print_info("\nCurrent Technical Indicators:")
        print_info(f"  RSI: {latest['RSI']:.2f}")
        print_info(f"  MACD: {latest['MACD']:.2f}")
        print_info(f"  Bollinger Width: {latest['BB_Width']:.4f}")
        print_info(f"  20-day Volatility: {latest['Volatility']:.4f}")
    elif error:
        print_warning(f"Historical data unavailable: {error.message}")

def demo_signal_generation(service, symbols):
    """Demonstrate signal generation with after-hours handling"""
    print_header("Signal Generation")
    
    for symbol in symbols:
        print(f"\n{Fore.MAGENTA}--- {symbol} ---{Style.RESET_ALL}")
        
        signal = service.generate_signal(symbol)
        
        if signal:
            # Determine signal color
            if signal.signal_type == "BUY":
                signal_color = Fore.GREEN
            elif signal.signal_type == "SELL":
                signal_color = Fore.RED
            else:
                signal_color = Fore.YELLOW
            
            print(f"{signal_color}Signal: {signal.signal_type}{Style.RESET_ALL}")
            print_info(f"Confidence: {signal.confidence:.2%}")
            print_info(f"Price Target: ${signal.price_target:.2f}")
            print_info(f"Stop Loss: ${signal.stop_loss:.2f}")
            print_info(f"Risk Score: {signal.risk_score:.2f}")
            
            if signal.is_after_hours:
                print_warning("Generated using after-hours/cached data")
            
            # Show key indicators
            if signal.indicators:
                print_info("Key Indicators:")
                for key in ["rsi", "macd", "volatility", "current_price"]:
                    if key in signal.indicators:
                        print_info(f"  {key}: {signal.indicators[key]:.2f}")

def demo_cache_behavior(service):
    """Demonstrate cache behavior"""
    print_header("Cache Behavior Demo")
    
    symbol = "MSFT"
    
    print_info("First fetch - populating cache...")
    tick1, error1 = service.fetch_real_time_data(symbol)
    if tick1:
        print_success(f"Fetched and cached: ${tick1.price:.2f}")
    
    print_info("\nSimulating market closed scenario...")
    # Force a market closed scenario by manipulating the check
    original_check = service.check_market_hours
    
    def mock_closed_hours():
        hours = original_check()
        hours.is_open = False
        hours.reason = "After-hours trading (simulated)"
        return hours
    
    service.check_market_hours = mock_closed_hours
    
    print_info("Second fetch - should use cache...")
    tick2, error2 = service.fetch_real_time_data(symbol)
    if tick2:
        print_success(f"Retrieved from cache: ${tick2.price:.2f}")
        if error2:
            print_warning(f"With warning: {error2.message}")
    
    # Restore original
    service.check_market_hours = original_check

def demo_error_scenarios(service):
    """Demonstrate various error scenarios"""
    print_header("Error Handling Scenarios")
    
    # Invalid symbol
    print_info("Testing invalid symbol...")
    tick, error = service.fetch_real_time_data("INVALID_XYZ")
    if error:
        print_warning(f"Error: {error.reason.value} - {error.message}")
        print_info(f"Recoverable: {error.is_recoverable}")
    
    # Test special invalid symbol
    print_info("\nTesting known invalid symbol...")
    signal = service.generate_signal("INVALID_SYMBOL")
    if signal:
        print_info(f"Signal: {signal.signal_type} (confidence: {signal.confidence:.2%})")

def main():
    """Main demonstration function"""
    print(f"{Fore.MAGENTA}🚀 After-Hours Data Handling Demo{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Using Mock Market Data Service{Style.RESET_ALL}")
    
    # Create service instance
    service = MockMarketDataService()
    
    # 1. Check market hours
    market_hours = demo_market_hours()
    
    # 2. Demonstrate data fetching
    demo_data_fetching(service, "AAPL")
    
    # 3. Show historical data
    demo_historical_data(service, "GOOGL")
    
    # 4. Generate signals for multiple symbols
    symbols = ["AAPL", "TSLA", "NVDA"]
    demo_signal_generation(service, symbols)
    
    # 5. Demonstrate cache behavior
    demo_cache_behavior(service)
    
    # 6. Show error handling
    demo_error_scenarios(service)
    
    # Summary
    print_header("Summary")
    
    if market_hours.is_open:
        print_info("✓ Market is currently open - live data would be fetched")
        print_info("✓ Data is cached for after-hours availability")
    else:
        print_info("✓ Market is closed - system uses intelligent fallbacks")
        print_info("✓ Cached data is used when available")
        print_info("✓ Appropriate warnings are provided")
    
    print_info("\nKey Features Demonstrated:")
    print_success("1. Automatic market hours detection")
    print_success("2. Smart error classification") 
    print_success("3. Cache fallback for after-hours")
    print_success("4. Confidence adjustment for stale data")
    print_success("5. Clear error messages and recovery suggestions")
    
    print(f"\n{Fore.GREEN}✨ The system ensures 24/7 availability for traders worldwide!{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 