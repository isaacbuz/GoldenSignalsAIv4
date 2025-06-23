"""
AI Chat API Endpoints

Provides REST API endpoints for the AI Trading Assistant chat functionality.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import logging

from src.services.ai_chat_service import (
    MultimodalAIChatService,
    ChatSession,
    AIResponse
)
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-chat", tags=["AI Chat"])

# Initialize AI chat service
ai_chat_service = None


def get_ai_chat_service(request: Request) -> MultimodalAIChatService:
    """Get AI chat service instance"""
    global ai_chat_service
    
    if ai_chat_service is None:
        config = {
            "openai_api_key": settings.openai_api_key if hasattr(settings, 'openai_api_key') else None,
            "anthropic_api_key": settings.anthropic_api_key if hasattr(settings, 'anthropic_api_key') else None,
        }
        ai_chat_service = MultimodalAIChatService(config)
        
        # Inject agent orchestrator if available
        if hasattr(request.app.state, 'agent_orchestrator'):
            ai_chat_service.agent_orchestrator = request.app.state.agent_orchestrator
    
    return ai_chat_service


@router.post("/sessions")
async def create_chat_session(
    request: Request,
    user_id: str = "default_user",
    context: Optional[dict] = None
):
    """Create a new chat session"""
    try:
        service = get_ai_chat_service(request)
        session = await service.create_session(user_id)
        
        # Add initial context if provided
        if context:
            session.context.update(context)
        
        return {
            "id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in session.messages
            ]
        }
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages")
async def send_message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    attachments: List[UploadFile] = File(None)
):
    """Send a message to the AI assistant with optional file attachments"""
    try:
        service = get_ai_chat_service(request)
        
        # Process the message
        response = await service.process_message(
            session_id=session_id,
            message=message,
            attachments=attachments
        )
        
        return {
            "message": response.message,
            "data": response.data,
            "visualizations": response.visualizations,
            "suggestions": response.suggestions,
            "confidence": response.confidence,
            "sources": response.sources
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(
    request: Request,
    session_id: str
):
    """Get chat session details"""
    try:
        service = get_ai_chat_service(request)
        
        if session_id not in service.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = service.sessions[session_id]
        
        return {
            "id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "attachments": msg.attachments,
                    "metadata": msg.metadata
                }
                for msg in session.messages[-50:]  # Last 50 messages
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    request: Request,
    session_id: str
):
    """Get a summary of the chat session"""
    try:
        service = get_ai_chat_service(request)
        summary = await service.get_session_summary(session_id)
        return summary
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(
    request: Request,
    session_id: str
):
    """Delete a chat session"""
    try:
        service = get_ai_chat_service(request)
        
        if session_id not in service.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del service.sessions[session_id]
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 