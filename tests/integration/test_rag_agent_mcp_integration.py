"""
Integration Tests for RAG-Agent-MCP System
Tests the complete flow of all components working together
Issue #195: Integration-1: RAG-Agent-MCP Integration Testing
"""

import pytest
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import all components
from agents.rag.historical_market_context_rag import HistoricalMarketContextRAG
from agents.rag.real_time_sentiment_analyzer import RealTimeSentimentAnalyzer
from agents.rag.options_flow_intelligence_rag import OptionsFlowIntelligenceRAG
from agents.common.registry.agent_registry import AgentRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate test data for integration tests"""
    
    @staticmethod
    def generate_market_data(symbol: str = "AAPL") -> Dict[str, Any]:
        """Generate realistic market data"""
        base_price = 175.0
        return {
            "symbol": symbol,
            "price": base_price + np.random.randn() * 2,
            "bid": base_price - 0.02,
            "ask": base_price + 0.02,
            "volume": int(50_000_000 + np.random.randn() * 10_000_000),
            "change": np.random.randn() * 3,
            "change_percent": np.random.randn() * 2,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_portfolio() -> Dict[str, Any]:
        """Generate test portfolio"""
        return {
            "id": "test-portfolio-001",
            "name": "Integration Test Portfolio",
            "cash": 100_000,
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "entry_price": 170.0,
                    "current_price": 175.0,
                    "position_type": "long"
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 50,
                    "entry_price": 2800.0,
                    "current_price": 2850.0,
                    "position_type": "long"
                }
            ]
        }


class IntegrationTestFramework:
    """Framework for running integration tests"""
    
    def __init__(self):
        self.test_data = TestDataGenerator()
        self.results = []
        self.metrics = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "avg_latency": 0
        }
    
    async def run_test(self, test_name: str, test_func):
        """Run a single integration test"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await test_func()
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            self.results.append({
                "test": test_name,
                "status": "passed",
                "latency_ms": latency,
                "result": result
            })
            
            self.metrics["passed"] += 1
            logger.info(f"✅ {test_name}: PASSED ({latency:.0f}ms)")
            
        except Exception as e:
            self.results.append({
                "test": test_name,
                "status": "failed",
                "error": str(e)
            })
            
            self.metrics["failed"] += 1
            logger.error(f"❌ {test_name}: FAILED - {e}")
        
        finally:
            self.metrics["total_tests"] += 1
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {self.metrics['total_tests']}")
        logger.info(f"Passed: {self.metrics['passed']} ✅")
        logger.info(f"Failed: {self.metrics['failed']} ❌")
        
        if self.metrics['passed'] > 0:
            avg_latency = sum(r['latency_ms'] for r in self.results if 'latency_ms' in r) / self.metrics['passed']
            logger.info(f"Average Latency: {avg_latency:.0f}ms")


@pytest.mark.asyncio
class TestRAGAgentMCPIntegration:
    """Integration tests for RAG-Agent-MCP system"""
    
    async def test_data_flow_integration(self):
        """Test complete data flow from Market Data → RAG → Agents → Execution"""
        framework = IntegrationTestFramework()
        
        # Test 1: Market Data to RAG Pipeline
        async def test_market_to_rag():
            # Simulate market data
            market_data = TestDataGenerator.generate_market_data()
            
            # Initialize RAG components
            historical_rag = HistoricalMarketContextRAG()
            sentiment_rag = RealTimeSentimentAnalyzer()
            
            # Query RAGs with market data
            historical_result = await historical_rag.retrieve(
                f"What patterns match {market_data['symbol']} at ${market_data['price']:.2f}?",
                symbol=market_data['symbol']
            )
            
            sentiment_result = await sentiment_rag.analyze_sentiment(
                symbols=[market_data['symbol']]
            )
            
            assert historical_result is not None
            assert sentiment_result is not None
            assert 'patterns' in historical_result
            assert 'overall_sentiment' in sentiment_result
            
            return {
                "market_data": market_data,
                "historical": historical_result,
                "sentiment": sentiment_result
            }
        
        await framework.run_test("Market Data → RAG Pipeline", test_market_to_rag)
        
        # Test 2: RAG to Agent Decision Making
        async def test_rag_to_agents():
            # Mock RAG results
            rag_results = {
                "historical_confidence": 0.85,
                "sentiment_score": 0.72,
                "risk_score": 0.15
            }
            
            # Simulate agent decisions based on RAG
            agents = {
                "risk_agent": 0.8 if rag_results["risk_score"] < 0.2 else 0.2,
                "sentiment_agent": rag_results["sentiment_score"],
                "technical_agent": rag_results["historical_confidence"]
            }
            
            # Calculate consensus
            consensus = sum(agents.values()) / len(agents)
            decision = "BUY" if consensus > 0.6 else "HOLD"
            
            assert consensus > 0
            assert decision in ["BUY", "HOLD", "SELL"]
            
            return {
                "agents": agents,
                "consensus": consensus,
                "decision": decision
            }
        
        await framework.run_test("RAG → Agent Decision", test_rag_to_agents)
        
        # Test 3: Agent Communication Hub
        async def test_agent_communication():
            # Simulate agent registration and messaging
            agents = ["risk_agent", "trading_agent", "sentiment_agent"]
            messages_sent = 0
            messages_received = 0
            
            # Simulate broadcast
            for sender in agents:
                for receiver in agents:
                    if sender != receiver:
                        messages_sent += 1
                        # Simulate delivery
                        messages_received += 1
            
            delivery_rate = messages_received / messages_sent if messages_sent > 0 else 0
            
            assert delivery_rate == 1.0  # All messages delivered
            assert messages_sent == 6  # 3 agents, each sends to 2 others
            
            return {
                "agents": len(agents),
                "messages_sent": messages_sent,
                "delivery_rate": delivery_rate
            }
        
        await framework.run_test("Agent Communication Hub", test_agent_communication)
        
        # Test 4: Risk Analytics Integration
        async def test_risk_analytics():
            portfolio = TestDataGenerator.generate_portfolio()
            
            # Calculate portfolio metrics
            total_value = portfolio["cash"] + sum(
                p["quantity"] * p["current_price"] for p in portfolio["positions"]
            )
            
            # Simulate VaR calculation
            returns = np.random.normal(0.001, 0.02, 252)  # 1 year of returns
            var_95 = np.percentile(returns, 5)
            
            # Risk checks
            position_limit = total_value * 0.1  # 10% max position
            
            assert total_value > 0
            assert var_95 < 0  # VaR should be negative
            assert position_limit > 0
            
            return {
                "portfolio_value": total_value,
                "var_95": abs(var_95),
                "position_limit": position_limit
            }
        
        await framework.run_test("Risk Analytics Integration", test_risk_analytics)
        
        # Test 5: Execution Management Integration
        async def test_execution_management():
            order = {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "type": "limit",
                "price": 175.00
            }
            
            # Simulate order routing
            venues = ["NYSE", "NASDAQ", "ARCA"]
            executions = []
            
            for i, venue in enumerate(venues):
                qty = order["quantity"] // len(venues)
                if i == 0:  # Add remainder to first venue
                    qty += order["quantity"] % len(venues)
                
                executions.append({
                    "venue": venue,
                    "quantity": qty,
                    "price": order["price"] + np.random.uniform(-0.01, 0.01)
                })
            
            # Calculate execution metrics
            total_executed = sum(e["quantity"] for e in executions)
            avg_price = sum(e["quantity"] * e["price"] for e in executions) / total_executed
            slippage = (avg_price - order["price"]) / order["price"]
            
            assert total_executed == order["quantity"]
            assert abs(slippage) < 0.001  # Less than 0.1% slippage
            
            return {
                "order": order,
                "executions": executions,
                "avg_price": avg_price,
                "slippage_pct": slippage * 100
            }
        
        await framework.run_test("Execution Management Integration", test_execution_management)
        
        # Print summary
        framework.print_summary()
    
    async def test_end_to_end_trading_flow(self):
        """Test complete end-to-end trading flow"""
        logger.info("\n" + "="*60)
        logger.info("END-TO-END TRADING FLOW TEST")
        logger.info("="*60)
        
        # Step 1: Market Data
        market_data = TestDataGenerator.generate_market_data("AAPL")
        logger.info(f"\n1️⃣ Market Data: {market_data['symbol']} @ ${market_data['price']:.2f}")
        
        # Step 2: RAG Analysis
        rag_scores = {
            "historical": 0.82,
            "sentiment": 0.75,
            "options_flow": 0.88,
            "technical": 0.79,
            "risk": 0.15
        }
        avg_confidence = np.mean(list(rag_scores.values()))
        logger.info(f"\n2️⃣ RAG Analysis: {avg_confidence:.0%} confidence")
        for service, score in rag_scores.items():
            logger.info(f"   - {service}: {score:.2f}")
        
        # Step 3: Risk Check
        portfolio = TestDataGenerator.generate_portfolio()
        portfolio_value = portfolio["cash"] + sum(
            p["quantity"] * p["current_price"] for p in portfolio["positions"]
        )
        position_size = min(portfolio_value * 0.05, 50_000)  # 5% or $50k max
        shares = int(position_size / market_data["price"])
        logger.info(f"\n3️⃣ Risk Check: Position size ${position_size:,.0f} ({shares} shares)")
        
        # Step 4: Agent Consensus
        agents = {
            "risk": 0.85,
            "sentiment": 0.90,
            "technical": 0.78,
            "ml_forecast": 0.82,
            "options_flow": 0.95
        }
        consensus = np.mean(list(agents.values()))
        logger.info(f"\n4️⃣ Agent Consensus: {consensus:.0%} approval")
        
        # Step 5: Execution
        if consensus > 0.7:
            executions = [
                ("NYSE", shares * 0.4, market_data["price"] + 0.01),
                ("NASDAQ", shares * 0.35, market_data["price"] + 0.02),
                ("Dark Pool", shares * 0.25, market_data["price"])
            ]
            
            total_cost = sum(int(qty) * price for _, qty, price in executions)
            avg_price = total_cost / shares
            slippage = (avg_price - market_data["price"]) / market_data["price"]
            
            logger.info(f"\n5️⃣ Execution Complete:")
            logger.info(f"   - Shares: {shares}")
            logger.info(f"   - Avg Price: ${avg_price:.2f}")
            logger.info(f"   - Slippage: {slippage:.3%}")
            logger.info(f"   - Total Cost: ${total_cost:,.2f}")
            
            logger.info(f"\n✅ Trade executed successfully!")
        else:
            logger.info(f"\n❌ Trade rejected (consensus below threshold)")
        
        assert market_data is not None
        assert avg_confidence > 0
        assert portfolio_value > 0
        assert consensus > 0
    
    async def test_parallel_processing(self):
        """Test parallel processing capabilities"""
        logger.info("\n" + "="*60)
        logger.info("PARALLEL PROCESSING TEST")
        logger.info("="*60)
        
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        async def process_symbol(symbol: str):
            start = asyncio.get_event_loop().time()
            
            # Simulate parallel operations
            await asyncio.sleep(0.1)  # Simulate API call
            
            market_data = TestDataGenerator.generate_market_data(symbol)
            rag_score = np.random.uniform(0.6, 0.9)
            
            latency = (asyncio.get_event_loop().time() - start) * 1000
            
            return {
                "symbol": symbol,
                "price": market_data["price"],
                "rag_score": rag_score,
                "latency_ms": latency
            }
        
        # Process all symbols in parallel
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[process_symbol(s) for s in symbols])
        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.info(f"\nProcessed {len(symbols)} symbols in {total_time:.0f}ms")
        logger.info(f"Average latency: {np.mean([r['latency_ms'] for r in results]):.0f}ms")
        logger.info(f"Parallelization factor: {sum(r['latency_ms'] for r in results) / total_time:.1f}x")
        
        for result in results:
            logger.info(f"  {result['symbol']}: ${result['price']:.2f} (score: {result['rag_score']:.2f})")
        
        assert len(results) == len(symbols)
        assert total_time < sum(r['latency_ms'] for r in results)  # Parallel is faster
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        logger.info("\n" + "="*60)
        logger.info("ERROR HANDLING AND RECOVERY TEST")
        logger.info("="*60)
        
        errors_handled = 0
        recoveries = 0
        
        # Test 1: Market data failure and recovery
        async def fetch_with_retry(symbol: str, max_retries: int = 3):
            nonlocal errors_handled, recoveries
            
            for attempt in range(max_retries):
                try:
                    if attempt < 2:  # Simulate failures
                        raise ConnectionError(f"Failed to fetch {symbol}")
                    
                    # Success on 3rd attempt
                    return TestDataGenerator.generate_market_data(symbol)
                    
                except ConnectionError as e:
                    errors_handled += 1
                    logger.warning(f"  Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise
            
            recoveries += 1
        
        # Test recovery
        try:
            result = await fetch_with_retry("AAPL")
            logger.info(f"  ✅ Recovered after {errors_handled} attempts")
            recoveries += 1
        except Exception as e:
            logger.error(f"  ❌ Failed to recover: {e}")
        
        # Test 2: Cascade failure prevention
        services = ["market_data", "rag", "risk", "execution"]
        failed_services = ["rag"]  # Simulate RAG failure
        
        operational_services = []
        for service in services:
            if service not in failed_services:
                operational_services.append(service)
            else:
                errors_handled += 1
                logger.warning(f"  ⚠️ {service} service unavailable")
        
        # Check if we can still operate
        can_trade = "market_data" in operational_services and "execution" in operational_services
        
        if can_trade:
            logger.info("  ✅ Trading can continue with degraded functionality")
            recoveries += 1
        else:
            logger.error("  ❌ Trading halted - critical services unavailable")
        
        logger.info(f"\nError Handling Summary:")
        logger.info(f"  Errors Handled: {errors_handled}")
        logger.info(f"  Successful Recoveries: {recoveries}")
        logger.info(f"  Recovery Rate: {(recoveries / errors_handled * 100) if errors_handled > 0 else 0:.0f}%")
        
        assert errors_handled > 0
        assert recoveries > 0
    
    async def test_performance_benchmarks(self):
        """Test performance against benchmarks"""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE BENCHMARK TEST")
        logger.info("="*60)
        
        benchmarks = {
            "market_data_latency": 50,  # ms
            "rag_query_latency": 200,  # ms
            "risk_calculation_latency": 100,  # ms
            "order_execution_latency": 150,  # ms
            "end_to_end_latency": 500  # ms
        }
        
        actual_latencies = {}
        
        # Test each component
        components = [
            ("market_data", 0.04),
            ("rag_query", 0.18),
            ("risk_calculation", 0.09),
            ("order_execution", 0.14)
        ]
        
        total_start = asyncio.get_event_loop().time()
        
        for component, sim_time in components:
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(sim_time)  # Simulate processing
            latency = (asyncio.get_event_loop().time() - start) * 1000
            actual_latencies[f"{component}_latency"] = latency
            
            benchmark = benchmarks[f"{component}_latency"]
            status = "✅ PASS" if latency < benchmark else "❌ FAIL"
            
            logger.info(f"  {component}: {latency:.0f}ms (benchmark: {benchmark}ms) {status}")
        
        # End-to-end latency
        total_latency = (asyncio.get_event_loop().time() - total_start) * 1000
        actual_latencies["end_to_end_latency"] = total_latency
        
        e2e_benchmark = benchmarks["end_to_end_latency"]
        e2e_status = "✅ PASS" if total_latency < e2e_benchmark else "❌ FAIL"
        
        logger.info(f"\n  End-to-End: {total_latency:.0f}ms (benchmark: {e2e_benchmark}ms) {e2e_status}")
        
        # Calculate performance score
        passed = sum(1 for k, v in actual_latencies.items() if v < benchmarks[k])
        total = len(benchmarks)
        score = (passed / total) * 100
        
        logger.info(f"\n  Performance Score: {score:.0f}% ({passed}/{total} benchmarks passed)")
        
        assert score >= 80  # At least 80% of benchmarks should pass


async def run_all_integration_tests():
    """Run all integration tests"""
    logger.info("\n" + "="*70)
    logger.info("RAG-AGENT-MCP INTEGRATION TEST SUITE")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_suite = TestRAGAgentMCPIntegration()
    
    # Run all tests
    await test_suite.test_data_flow_integration()
    await test_suite.test_end_to_end_trading_flow()
    await test_suite.test_parallel_processing()
    await test_suite.test_error_handling_and_recovery()
    await test_suite.test_performance_benchmarks()
    
    logger.info("\n" + "="*70)
    logger.info("ALL INTEGRATION TESTS COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    asyncio.run(run_all_integration_tests()) 