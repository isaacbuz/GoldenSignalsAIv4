#!/usr/bin/env python3
"""
Start Live Data Feed for GoldenSignalsAI V2
Connects to real market data and feeds all agents
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data components
from infrastructure.data.fetchers.live_data_fetcher import (
    UnifiedDataFeed, 
    YahooFinanceSource, 
    PolygonIOSource,
    AgentDataAdapter,
    MarketData,
    OptionsData
)

# Import agent components
from agents.common.data_bus import AgentDataBus
from agents.orchestration.hybrid_orchestrator import HybridOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """Main system that connects live data to agents"""
    
    def __init__(self):
        self.data_bus = AgentDataBus()
        self.data_feed = UnifiedDataFeed(primary_source="yahoo")
        self.adapter = AgentDataAdapter(self.data_bus)
        self.orchestrator = HybridOrchestrator(
            symbols=['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ'],
            data_bus=self.data_bus
        )
        self.running = False
        self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'META']
        
    async def setup_data_sources(self):
        """Set up data sources based on available API keys"""
        # Always add Yahoo Finance (free)
        yahoo_source = YahooFinanceSource()
        self.data_feed.add_source("yahoo", yahoo_source)
        logger.info("✅ Added Yahoo Finance data source")
        
        # Add Polygon if API key is available
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key:
            polygon_source = PolygonIOSource(polygon_key)
            self.data_feed.add_source("polygon", polygon_source)
            logger.info("✅ Added Polygon.io data source")
        else:
            logger.warning("⚠️ POLYGON_API_KEY not found - using free data only")
            
        # Register data callbacks
        self.data_feed.register_callback('quote', self.on_quote_update)
        self.data_feed.register_callback('options', self.on_options_update)
        
    async def on_quote_update(self, data: MarketData):
        """Handle real-time quote updates"""
        # Update adapter (which publishes to data bus)
        await self.adapter.on_quote_update(data)
        
        # Log price update
        logger.info(
            f"📊 {data.symbol}: ${data.price:.2f} "
            f"(Bid: ${data.bid:.2f}, Ask: ${data.ask:.2f}, "
            f"Volume: {data.volume:,})"
        )
        
    async def on_options_update(self, options: list[OptionsData]):
        """Handle options chain updates"""
        if not options:
            return
            
        # Update adapter
        await self.adapter.on_options_update(options)
        
        # Log summary
        symbol = options[0].symbol
        calls = len([opt for opt in options if opt.option_type == 'call'])
        puts = len([opt for opt in options if opt.option_type == 'put'])
        
        logger.info(
            f"📈 {symbol} Options: {calls} calls, {puts} puts updated"
        )
        
    async def process_signals(self):
        """Process signals from orchestrator"""
        while self.running:
            try:
                # Run orchestrator analysis on all symbols
                for symbol in self.symbols:
                    # Check if we have data for this symbol
                    if symbol in self.adapter.price_history:
                        signals = await self.orchestrator.analyze_symbol(symbol)
                        
                        if signals:
                            self.display_signals(symbol, signals)
                            
                # Wait before next analysis
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Error processing signals: {e}")
                await asyncio.sleep(5)
                
    def display_signals(self, symbol: str, signals: Dict[str, Any]):
        """Display trading signals"""
        print(f"\n{'='*60}")
        print(f"🎯 TRADING SIGNALS FOR {symbol}")
        print(f"{'='*60}")
        
        # Final signal
        final_signal = signals.get('final_signal', {})
        if final_signal:
            action = final_signal.get('action', 'HOLD')
            confidence = final_signal.get('confidence', 0) * 100
            
            # Color coding
            if action == 'BUY':
                print(f"✅ ACTION: {action} (Confidence: {confidence:.1f}%)")
            elif action == 'SELL':
                print(f"🔴 ACTION: {action} (Confidence: {confidence:.1f}%)")
            else:
                print(f"⏸️ ACTION: {action} (Confidence: {confidence:.1f}%)")
                
        # Sentiment
        sentiment = signals.get('sentiment_analysis', {})
        if sentiment:
            print(f"\n📊 Market Sentiment:")
            print(f"   Independent: {sentiment.get('independent_sentiment', 'neutral')}")
            print(f"   Collaborative: {sentiment.get('collaborative_sentiment', 'neutral')}")
            print(f"   Confidence: {sentiment.get('sentiment_confidence', 0)*100:.1f}%")
            
        # Divergences
        divergences = signals.get('divergence_analysis', {})
        if divergences.get('has_divergence'):
            print(f"\n⚠️ DIVERGENCE DETECTED!")
            print(f"   Type: {divergences.get('divergence_type', 'unknown')}")
            print(f"   Score: {divergences.get('divergence_score', 0):.2f}")
            
        # Top agents
        agent_results = signals.get('agent_results', [])
        if agent_results:
            print(f"\n🤖 Top Contributing Agents:")
            for i, agent in enumerate(agent_results[:3]):
                print(f"   {i+1}. {agent['agent']}: {agent['signal']} ({agent['confidence']*100:.1f}%)")
                
        print(f"{'='*60}\n")
        
    async def start(self):
        """Start the live trading system"""
        try:
            logger.info("🚀 Starting GoldenSignalsAI Live Trading System...")
            
            # Set up data sources
            await self.setup_data_sources()
            
            # Connect all sources
            await self.data_feed.connect_all()
            
            # Initialize orchestrator
            await self.orchestrator.initialize()
            
            self.running = True
            
            # Start tasks
            tasks = [
                self.data_feed.start_streaming(self.symbols, interval=5),  # 5 second updates
                self.process_signals()
            ]
            
            logger.info(f"📡 Streaming live data for: {', '.join(self.symbols)}")
            logger.info("💡 Signals will be generated every 30 seconds")
            logger.info("Press Ctrl+C to stop...")
            
            # Run all tasks
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.running = False
            self.data_feed.stop_streaming()
            await self.orchestrator.shutdown()
            
async def main():
    """Main entry point"""
    system = LiveTradingSystem()
    await system.start()
    
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          GoldenSignalsAI V2 - Live Trading System         ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Connecting to real market data and running AI agents...  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check for API keys
    if not os.getenv('POLYGON_API_KEY'):
        print("ℹ️ TIP: Set POLYGON_API_KEY environment variable for professional data")
        print("   export POLYGON_API_KEY='your_api_key_here'")
        print("   Using free Yahoo Finance data for now...\n")
        
    asyncio.run(main()) 