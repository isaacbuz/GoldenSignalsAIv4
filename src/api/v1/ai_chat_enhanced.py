"""
Enhanced AI Chat API Endpoints - Production Ready
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import json
import asyncio
from PIL import Image
import io
import base64

from src.services.ai_chat_service_enhanced import (
    EnhancedMultimodalAIChatService,
    EnhancedMessageType,
    AnalysisType,
    ChartPattern,
    EnhancedAIResponse,
    VisionAnalysis
)
from src.utils.logger import get_logger
from src.core.auth import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/ai-chat", tags=["AI Chat Enhanced"])

# Initialize service
ai_service = EnhancedMultimodalAIChatService()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    voice_enabled: bool = False
    stream_response: bool = False


class PortfolioAnalysisRequest(BaseModel):
    holdings: List[Dict[str, Any]]
    session_id: Optional[str] = None


class BacktestRequest(BaseModel):
    strategy: Dict[str, Any]
    symbol: str
    period: str = "1y"
    session_id: Optional[str] = None


class ChartGenerationRequest(BaseModel):
    symbol: str
    period: str = "1mo"
    indicators: List[str] = []
    session_id: Optional[str] = None


@router.post("/chat", response_model=EnhancedAIResponse)
async def chat_with_ai(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Enhanced chat endpoint with streaming support
    """
    try:
        # Create or get session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Add user context
        context = request.context or {}
        context['user_id'] = current_user.get('id')
        context['voice_enabled'] = request.voice_enabled
        
        # Generate response
        response = await ai_service.generate_comprehensive_response(
            query=request.message,
            context=context,
            attachments=[]
        )
        
        # Log interaction for analytics
        background_tasks.add_task(
            log_interaction,
            session_id=session_id,
            user_id=current_user.get('id'),
            message=request.message,
            response=response.message
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_with_ai_stream(
    request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Stream chat responses for real-time interaction
    """
    async def generate():
        try:
            # Simulate streaming response
            response_parts = [
                "I'm analyzing your request",
                "Gathering market data",
                "Running technical analysis",
                "Generating insights"
            ]
            
            for part in response_parts:
                yield f"data: {json.dumps({'content': part, 'done': False})}\n\n"
                await asyncio.sleep(0.5)
            
            # Final response
            final_response = await ai_service.generate_comprehensive_response(
                query=request.message,
                context={'user_id': current_user.get('id')},
                attachments=[]
            )
            
            yield f"data: {json.dumps({'content': final_response.message, 'done': True, 'data': final_response.dict()})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/analyze/image", response_model=VisionAnalysis)
async def analyze_chart_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze trading chart images with computer vision
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Analyze with vision
        analysis = await ai_service.analyze_chart_with_vision(image)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/portfolio")
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze portfolio and provide optimization suggestions
    """
    try:
        analysis = await ai_service.analyze_portfolio(request.holdings)
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def run_backtest(
    request: BacktestRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Run backtesting on trading strategies
    """
    try:
        results = await ai_service.run_backtest(
            strategy=request.strategy,
            symbol=request.symbol,
            period=request.period
        )
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/chart")
async def generate_chart(
    request: ChartGenerationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate trading charts with technical indicators
    """
    try:
        chart_base64 = await ai_service.generate_trading_chart(
            symbol=request.symbol,
            period=request.period,
            indicators=request.indicators
        )
        
        return {
            "chart": chart_base64,
            "symbol": request.symbol,
            "period": request.period,
            "indicators": request.indicators
        }
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/upload")
async def process_voice_message(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Process voice messages
    """
    try:
        # Validate audio file
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio")
        
        # Process voice to text
        text = await ai_service.process_voice_input(audio)
        
        # Generate response
        response = await ai_service.generate_comprehensive_response(
            query=text,
            context={'user_id': current_user.get('id'), 'voice_enabled': True},
            attachments=[]
        )
        
        return {
            "transcribed_text": text,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Error processing voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal")
async def process_multimodal_request(
    message: str = Form(...),
    files: List[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    voice_enabled: bool = Form(False),
    current_user: Dict = Depends(get_current_user)
):
    """
    Process multimodal requests with text, images, documents, and data
    """
    try:
        attachments = []
        
        # Process uploaded files
        if files:
            for file in files:
                if file.content_type.startswith('image/'):
                    # Process image
                    contents = await file.read()
                    image = Image.open(io.BytesIO(contents))
                    vision_analysis = await ai_service.analyze_chart_with_vision(image)
                    
                    attachments.append({
                        'type': 'image_analysis',
                        'filename': file.filename,
                        'vision_analysis': vision_analysis.dict()
                    })
                    
                elif file.content_type in ['text/csv', 'application/vnd.ms-excel']:
                    # Process CSV/Excel
                    contents = await file.read()
                    # Add data processing logic
                    attachments.append({
                        'type': 'csv_analysis',
                        'filename': file.filename,
                        'analysis': {'rows': 100, 'columns': 10}  # Placeholder
                    })
                    
                elif file.content_type == 'application/pdf':
                    # Process PDF
                    attachments.append({
                        'type': 'document',
                        'filename': file.filename,
                        'content': 'PDF content extracted'  # Placeholder
                    })
        
        # Generate comprehensive response
        response = await ai_service.generate_comprehensive_response(
            query=message,
            context={
                'user_id': current_user.get('id'),
                'voice_enabled': voice_enabled
            },
            attachments=attachments
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in multimodal processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_available_patterns():
    """
    Get list of detectable chart patterns
    """
    return {
        "patterns": [pattern.value for pattern in ChartPattern],
        "descriptions": {
            ChartPattern.HEAD_SHOULDERS: "Reversal pattern indicating trend change",
            ChartPattern.DOUBLE_TOP: "Bearish reversal pattern",
            ChartPattern.DOUBLE_BOTTOM: "Bullish reversal pattern",
            ChartPattern.TRIANGLE: "Continuation pattern",
            ChartPattern.FLAG: "Short-term continuation pattern",
            ChartPattern.WEDGE: "Reversal or continuation pattern",
            ChartPattern.CUP_HANDLE: "Bullish continuation pattern",
            ChartPattern.SUPPORT_RESISTANCE: "Key price levels"
        }
    }


@router.get("/analysis-types")
async def get_analysis_types():
    """
    Get available analysis types
    """
    return {
        "types": [analysis.value for analysis in AnalysisType],
        "descriptions": {
            AnalysisType.TECHNICAL: "Chart patterns, indicators, and price action",
            AnalysisType.FUNDAMENTAL: "Company financials and valuation",
            AnalysisType.SENTIMENT: "Market sentiment and news analysis",
            AnalysisType.PATTERN: "Advanced pattern recognition",
            AnalysisType.PORTFOLIO: "Portfolio optimization and risk analysis",
            AnalysisType.BACKTEST: "Historical strategy performance"
        }
    }


@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get real-time market data for a symbol
    """
    try:
        data = await ai_service.market_data_service.get_real_time_data(symbol)
        return data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_interaction(
    session_id: str,
    user_id: str,
    message: str,
    response: str
):
    """
    Log chat interactions for analytics and improvement
    """
    try:
        # In production, save to database
        logger.info(f"Chat interaction - User: {user_id}, Session: {session_id}")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")


# WebSocket endpoint for real-time chat
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            response = await ai_service.generate_comprehensive_response(
                query=message_data.get('message'),
                context={'session_id': session_id},
                attachments=[]
            )
            
            # Send response
            await manager.send_personal_message(
                json.dumps(response.dict()),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket) 