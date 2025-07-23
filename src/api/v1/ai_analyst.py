"""
AI Trading Analyst API endpoints
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from src.services.ai_trading_analyst import AITradingAnalyst
from src.services.chart_vision_analyzer import ChartVisionAnalyzer, analyze_chart_screenshot
from src.services.websocket_manager import WebSocketManager

router = APIRouter(prefix="/ai-analyst", tags=["AI Analyst"])

# Initialize services
ai_analyst = AITradingAnalyst()
ws_manager = WebSocketManager()
chart_analyzer = ChartVisionAnalyzer()


@router.post("/analyze")
async def analyze_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a trading query and return comprehensive analysis
    """
    try:
        # Process the query
        response = await ai_analyst.analyze_query(query, context)
        
        # Convert response to dict
        return {
            'status': 'success',
            'analysis': response.text_analysis,
            'charts': response.charts,
            'insights': response.key_insights,
            'recommendations': response.recommendations,
            'confidence': response.confidence_score,
            'data_tables': response.data_tables,
            'alerts': response.alerts,
            'follow_up_questions': response.follow_up_questions,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/technical")
async def analyze_technical(
    symbol: str,
    timeframe: str = '1h',
    indicators: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform technical analysis on a symbol
    """
    query = f"Perform technical analysis on {symbol} {timeframe} chart"
    if indicators:
        query += f" with {', '.join(indicators)}"
    
    context = {
        'analysis_type': 'technical',
        'symbol': symbol,
        'timeframe': timeframe,
        'indicators': indicators or []
    }
    
    return await analyze_query(query, context)


@router.post("/analyze/sentiment")
async def analyze_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Analyze market sentiment for a symbol
    """
    query = f"What is the market sentiment for {symbol}?"
    context = {
        'analysis_type': 'sentiment',
        'symbol': symbol
    }
    
    return await analyze_query(query, context)


@router.post("/analyze/patterns")
async def analyze_patterns(
    symbol: str,
    timeframe: str = '1h',
    pattern_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Detect patterns for a symbol
    """
    query = f"Find chart patterns on {symbol} {timeframe}"
    if pattern_types:
        query += f" focusing on {', '.join(pattern_types)}"
    
    context = {
        'analysis_type': 'patterns',
        'symbol': symbol,
        'timeframe': timeframe,
        'pattern_types': pattern_types or []
    }
    
    return await analyze_query(query, context)


@router.post("/analyze/risk")
async def analyze_risk(
    symbol: str,
    position_type: Optional[str] = None,
    position_size: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze risk for a position
    """
    query = f"Analyze risk for {symbol}"
    if position_type:
        query += f" {position_type} position"
    
    context = {
        'analysis_type': 'risk',
        'symbol': symbol,
        'position_type': position_type,
        'position_size': position_size
    }
    
    return await analyze_query(query, context)


@router.post("/analyze/compare")
async def compare_symbols(
    symbols: List[str],
    metrics: Optional[List[str]] = None,
    timeframe: str = '1d'
) -> Dict[str, Any]:
    """
    Compare multiple symbols
    """
    query = f"Compare {' vs '.join(symbols)}"
    if metrics:
        query += f" on {', '.join(metrics)}"
    
    context = {
        'analysis_type': 'comparison',
        'symbols': symbols,
        'metrics': metrics or ['performance', 'volatility', 'momentum'],
        'timeframe': timeframe
    }
    
    return await analyze_query(query, context)


@router.post("/analyze/prediction")
async def predict_price(
    symbol: str,
    timeframe: str = '1d',
    horizon: str = '1w'
) -> Dict[str, Any]:
    """
    Predict price movement
    """
    query = f"Predict where {symbol} will be in {horizon}"
    context = {
        'analysis_type': 'prediction',
        'symbol': symbol,
        'timeframe': timeframe,
        'prediction_horizon': horizon
    }
    
    return await analyze_query(query, context)


@router.post("/analyze/chart-image")
async def analyze_chart_image(
    image_data: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a chart screenshot using computer vision
    
    Args:
        image_data: Base64 encoded image data
        context: Optional context about what to analyze (e.g., "looking for entry points")
    
    Returns:
        Comprehensive chart analysis with patterns, levels, and trading signals
    """
    try:
        # Analyze the chart image
        result = await chart_analyzer.analyze_chart_image(image_data, context)
        
        return {
            'status': 'success',
            'analysis': result['analysis'],
            'visual_data': result['visual_data'],
            'trading_signals': result['trading_signals'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart analysis error: {str(e)}")


@router.websocket("/stream")
async def analyst_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI analyst interaction
    """
    await websocket.accept()
    connection_id = await ws_manager.connect(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            'type': 'connection',
            'status': 'connected',
            'connection_id': connection_id,
            'message': 'AI Trading Analyst ready'
        })
        
        # Initialize session context
        session_context = {
            'connection_id': connection_id,
            'recent_symbols': [],
            'current_timeframe': '1h',
            'active_indicators': [],
            'conversation_history': []
        }
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get('type') == 'query':
                query = data.get('query', '')
                
                # Update session context
                session_context['conversation_history'].append({
                    'role': 'user',
                    'content': query,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Send thinking status
                await websocket.send_json({
                    'type': 'status',
                    'message': 'Analyzing your query...'
                })
                
                try:
                    # Analyze query
                    response = await ai_analyst.analyze_query(query, session_context)
                    
                    # Update context with extracted entities
                    if response.key_insights:
                        # Extract symbols from analysis
                        # This would be more sophisticated in practice
                        pass
                    
                    # Send analysis response
                    await websocket.send_json({
                        'type': 'analysis',
                        'analysis': response.text_analysis,
                        'charts': response.charts,
                        'insights': response.key_insights,
                        'recommendations': response.recommendations,
                        'confidence': response.confidence_score,
                        'data_tables': response.data_tables,
                        'alerts': response.alerts,
                        'follow_up_questions': response.follow_up_questions,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Update conversation history
                    session_context['conversation_history'].append({
                        'role': 'assistant',
                        'content': response.text_analysis,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': f'Analysis error: {str(e)}'
                    })
            
            elif data.get('type') == 'subscribe':
                # Subscribe to real-time updates for a symbol
                symbol = data.get('symbol')
                if symbol:
                    # This would set up real-time monitoring
                    await websocket.send_json({
                        'type': 'subscription',
                        'status': 'subscribed',
                        'symbol': symbol,
                        'message': f'Subscribed to real-time updates for {symbol}'
                    })
            
            elif data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
            
            elif data.get('type') == 'chart_image':
                # Handle chart image analysis
                image_data = data.get('image')
                context = data.get('context')
                
                await websocket.send_json({
                    'type': 'status',
                    'message': 'Analyzing your chart...'
                })
                
                try:
                    # Analyze chart
                    result = await chart_analyzer.analyze_chart_image(image_data, context)
                    
                    # Send analysis response
                    await websocket.send_json({
                        'type': 'chart_analysis',
                        'analysis': result['analysis'],
                        'visual_data': result['visual_data'],
                        'trading_signals': result['trading_signals'],
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': f'Chart analysis error: {str(e)}'
                    })
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(connection_id)
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
        await ws_manager.disconnect(connection_id)


@router.get("/suggestions")
async def get_query_suggestions(partial_query: str) -> List[str]:
    """
    Get query suggestions based on partial input
    """
    suggestions = [
        f"Analyze {partial_query} technical setup",
        f"What's the sentiment for {partial_query}?",
        f"Find patterns on {partial_query}",
        f"Compare {partial_query} with sector peers",
        f"Predict {partial_query} price movement",
        f"Show {partial_query} options flow",
        f"Risk analysis for {partial_query}",
        f"Is {partial_query} a good buy?"
    ]
    
    # Filter suggestions based on partial query
    filtered = [s for s in suggestions if partial_query.lower() in s.lower()]
    
    return filtered[:5]  # Return top 5 suggestions


@router.get("/examples")
async def get_example_queries() -> List[Dict[str, str]]:
    """
    Get example queries for users
    """
    return [
        {
            "category": "Technical Analysis",
            "queries": [
                "Analyze AAPL technical setup on the daily chart",
                "Show me SPY with RSI and MACD indicators",
                "What are the support and resistance levels for TSLA?"
            ]
        },
        {
            "category": "Pattern Recognition",
            "queries": [
                "Find head and shoulders pattern on NVDA",
                "Detect chart patterns on QQQ hourly chart",
                "Show me triangle patterns in the market"
            ]
        },
        {
            "category": "Market Sentiment",
            "queries": [
                "What's the sentiment for TSLA?",
                "How is the market feeling about tech stocks?",
                "Show me options flow sentiment for SPY"
            ]
        },
        {
            "category": "Predictions",
            "queries": [
                "Predict where AAPL will be next week",
                "What's the price target for MSFT?",
                "Will SPY break 450 this month?"
            ]
        },
        {
            "category": "Comparisons",
            "queries": [
                "Compare NVDA vs AMD performance",
                "How does AAPL compare to other FAANG stocks?",
                "Show me semiconductor sector comparison"
            ]
        },
        {
            "category": "Risk Analysis",
            "queries": [
                "What's the risk of holding SPY calls?",
                "Analyze downside risk for my TSLA position",
                "Calculate position size for AAPL trade"
            ]
        }
    ]


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "AI Trading Analyst",
        "timestamp": datetime.now().isoformat()
    } 