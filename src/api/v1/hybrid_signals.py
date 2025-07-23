"""
Hybrid Signal System API Endpoints
Provides access to the enhanced hybrid sentiment trading system
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.orchestration.hybrid_orchestrator import HybridOrchestrator
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/hybrid", tags=["hybrid"])

# Global orchestrator instance
orchestrator = None

def get_orchestrator() -> HybridOrchestrator:
    """Get or create orchestrator instance"""
    global orchestrator
    if not orchestrator:
        # Default symbols - can be configured
        default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'SPY', 'QQQ']
        orchestrator = HybridOrchestrator(symbols=default_symbols)
        logger.info(f"Created hybrid orchestrator with {len(default_symbols)} symbols")
    return orchestrator

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check health of hybrid system"""
    try:
        orch = get_orchestrator()
        return {
            "status": "healthy",
            "agents": len(orch.agents),
            "symbols": len(orch.symbols),
            "data_bus": "active" if orch.data_bus else "inactive",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals/{symbol}")
async def get_signals(symbol: str) -> Dict[str, Any]:
    """Get hybrid signals for a specific symbol"""
    try:
        orch = get_orchestrator()
        
        # Add symbol if not already monitored
        if symbol not in orch.symbols:
            orch.symbols.append(symbol)
            logger.info(f"Added {symbol} to monitored symbols")
        
        # Generate signals
        signal = orch.generate_signals_for_symbol(symbol)
        
        # Add API metadata
        signal['api_metadata'] = {
            'endpoint': 'hybrid_signals',
            'version': '2.0',
            'enhanced_features': [
                'sentiment_analysis',
                'divergence_detection',
                'collaborative_intelligence',
                'performance_tracking'
            ]
        }
        
        return signal
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals")
async def get_all_signals(
    symbols: Optional[List[str]] = Query(None, description="List of symbols to analyze")
) -> Dict[str, Any]:
    """Get signals for multiple symbols"""
    try:
        orch = get_orchestrator()
        
        # Use provided symbols or default
        target_symbols = symbols if symbols else orch.symbols[:5]  # Limit to 5 for performance
        
        # Generate signals concurrently
        async def get_signal_async(symbol):
            return await asyncio.to_thread(orch.generate_signals_for_symbol, symbol)
        
        tasks = [get_signal_async(symbol) for symbol in target_symbols]
        signals = await asyncio.gather(*tasks)
        
        # Compile results
        results = {
            "signals": {signal['symbol']: signal for signal in signals},
            "summary": {
                "total_signals": len(signals),
                "buy_signals": sum(1 for s in signals if s['action'] == 'BUY'),
                "sell_signals": sum(1 for s in signals if s['action'] == 'SELL'),
                "hold_signals": sum(1 for s in signals if s['action'] == 'HOLD'),
                "average_confidence": sum(s['confidence'] for s in signals) / len(signals) if signals else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating bulk signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Get sentiment analysis for symbol or market"""
    try:
        orch = get_orchestrator()
        
        # Get sentiment analysis
        sentiment = orch.get_sentiment_analysis(symbol)
        
        # Add visualization-friendly data
        sentiment['visualization'] = {
            'sentiment_distribution': {
                'bullish': sentiment['market_sentiment']['breakdown']['bullish'],
                'bearish': sentiment['market_sentiment']['breakdown']['bearish'],
                'neutral': sentiment['market_sentiment']['breakdown']['neutral']
            },
            'divergence_rate': sum(
                1 for data in sentiment['agent_sentiments'].values()
                if data['current'].get('divergence', False)
            ) / len(sentiment['agent_sentiments']) if sentiment['agent_sentiments'] else 0
        }
        
        return sentiment
        
    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_dashboard() -> Dict[str, Any]:
    """Get comprehensive performance dashboard"""
    try:
        orch = get_orchestrator()
        dashboard = orch.get_performance_dashboard()
        
        # Add additional metrics
        dashboard['enhanced_metrics'] = {
            'system_health': _calculate_system_health(dashboard),
            'top_performers': _get_top_performers(dashboard),
            'improvement_suggestions': _generate_suggestions(dashboard)
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/update")
async def update_performance(
    agent_name: str,
    signal_id: str,
    outcome: float
) -> Dict[str, Any]:
    """Update agent performance with signal outcome"""
    try:
        orch = get_orchestrator()
        
        # Validate outcome
        if not -1.0 <= outcome <= 1.0:
            raise ValueError("Outcome must be between -1.0 and 1.0")
        
        # Update performance
        orch.update_agent_performance(agent_name, signal_id, outcome)
        
        # Get updated metrics
        if agent_name in orch.agents:
            agent = orch.agents[agent_name]
            metrics = agent.get_performance_metrics()
            
            return {
                "status": "success",
                "agent": agent_name,
                "updated_metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise ValueError(f"Agent {agent_name} not found")
            
    except Exception as e:
        logger.error(f"Error updating performance: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """List all active hybrid agents"""
    try:
        orch = get_orchestrator()
        
        agents_info = {}
        for name, agent in orch.agents.items():
            metrics = agent.get_performance_metrics()
            agents_info[name] = {
                "type": "hybrid",
                "base_indicator": agent.base_indicator,
                "performance": {
                    "independent_accuracy": metrics['performance']['independent']['accuracy'],
                    "collaborative_accuracy": metrics['performance']['collaborative']['accuracy'],
                    "divergence_rate": metrics['divergence_rate']
                },
                "current_weights": metrics['current_weights'],
                "status": "active"
            }
        
        return {
            "agents": agents_info,
            "total_count": len(agents_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/divergences")
async def get_divergence_analysis(
    min_count: int = Query(1, description="Minimum divergence count")
) -> Dict[str, Any]:
    """Get analysis of current divergences across all symbols"""
    try:
        orch = get_orchestrator()
        
        divergence_data = {
            "symbols_with_divergences": [],
            "total_divergences": 0,
            "divergence_patterns": {},
            "opportunities": []
        }
        
        # Analyze each symbol
        for symbol in orch.symbols[:10]:  # Limit to prevent timeout
            signal = orch.generate_signals_for_symbol(symbol)
            div_analysis = signal['metadata']['divergence_analysis']
            
            if div_analysis['count'] >= min_count:
                divergence_data['symbols_with_divergences'].append({
                    'symbol': symbol,
                    'count': div_analysis['count'],
                    'strong_count': len(div_analysis['strong_divergences']),
                    'action': signal['action'],
                    'confidence': signal['confidence']
                })
                
                divergence_data['total_divergences'] += div_analysis['count']
                
                # Track patterns
                for div in div_analysis['strong_divergences']:
                    pattern = f"{div['independent']}_vs_{div['collaborative']}"
                    if pattern not in divergence_data['divergence_patterns']:
                        divergence_data['divergence_patterns'][pattern] = 0
                    divergence_data['divergence_patterns'][pattern] += 1
                
                # Identify opportunities
                if div_analysis['opportunities']:
                    divergence_data['opportunities'].append({
                        'symbol': symbol,
                        'opportunities': div_analysis['opportunities'],
                        'confidence': signal['confidence']
                    })
        
        # Sort by divergence count
        divergence_data['symbols_with_divergences'].sort(
            key=lambda x: x['count'], 
            reverse=True
        )
        
        return divergence_data
        
    except Exception as e:
        logger.error(f"Error analyzing divergences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

def _calculate_system_health(dashboard: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall system health metrics"""
    market_perf = dashboard['market_performance']
    
    # Health score based on various factors
    health_score = 100.0
    
    # Penalize high divergence rate
    if market_perf['average_divergence_rate'] > 0.3:
        health_score -= 10
    elif market_perf['average_divergence_rate'] > 0.5:
        health_score -= 20
    
    # Reward consistent performance
    avg_accuracy = sum(
        agent['performance']['overall']['accuracy'] 
        for agent in dashboard['agents'].values()
    ) / len(dashboard['agents']) if dashboard['agents'] else 0.5
    
    health_score += (avg_accuracy - 0.5) * 40  # +/- 20 points based on accuracy
    
    return {
        "score": max(0, min(100, health_score)),
        "status": "excellent" if health_score > 80 else "good" if health_score > 60 else "needs_attention",
        "factors": {
            "divergence_impact": market_perf['average_divergence_rate'],
            "accuracy_impact": avg_accuracy
        }
    }

def _get_top_performers(dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify top performing agents"""
    performers = []
    
    for agent_name, data in dashboard['agents'].items():
        overall_accuracy = data['performance']['overall']['accuracy']
        divergence_bonus = data['performance'].get('divergence_bonus', 0)
        
        # Combined score
        score = overall_accuracy + divergence_bonus * 0.1
        
        performers.append({
            'agent': agent_name,
            'score': score,
            'accuracy': overall_accuracy,
            'divergence_rate': data['divergence_rate']
        })
    
    # Sort and return top 5
    performers.sort(key=lambda x: x['score'], reverse=True)
    return performers[:5]

def _generate_suggestions(dashboard: Dict[str, Any]) -> List[str]:
    """Generate improvement suggestions based on performance"""
    suggestions = []
    
    # Check divergence rate
    if dashboard['market_performance']['average_divergence_rate'] > 0.4:
        suggestions.append("High divergence rate detected. Consider reviewing agent correlations and data sources.")
    
    # Check for underperforming agents
    for agent_name, data in dashboard['agents'].items():
        if data['performance']['overall']['accuracy'] < 0.4:
            suggestions.append(f"{agent_name} showing low accuracy. Consider retraining or adjusting parameters.")
    
    # Check sentiment consistency
    if dashboard['market_performance'].get('sentiment_volatility', 0) > 0.3:
        suggestions.append("High sentiment volatility. Market conditions may be uncertain.")
    
    if not suggestions:
        suggestions.append("System performing well. Continue monitoring for optimization opportunities.")
    
    return suggestions

# WebSocket endpoint for real-time updates
@router.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket for real-time signal updates"""
    await websocket.accept()
    try:
        orch = get_orchestrator()
        
        while True:
            # Send updates every 5 seconds
            for symbol in orch.symbols[:3]:  # Limit for performance
                signal = orch.generate_signals_for_symbol(symbol)
                await websocket.send_json({
                    "type": "signal_update",
                    "data": signal
                })
            
            # Send sentiment update
            sentiment = orch.get_sentiment_analysis()
            await websocket.send_json({
                "type": "sentiment_update",
                "data": sentiment['market_sentiment']
            })
            
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close() 