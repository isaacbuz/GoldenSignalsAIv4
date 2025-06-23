from fastapi import APIRouter, Request
from agents.meta.feedback_analyst_agent import FeedbackAnalystAgent

router = APIRouter()

@router.post("/api/feedback")
async def feedback_llm(request: Request):
    payload = await request.json()
    logs = payload.get("feedback_logs", [])
    agent_name = payload.get("agent_name", None)
    use_grok = payload.get("use_grok", False)
    analyst = FeedbackAnalystAgent(use_grok=use_grok)
    summary = analyst.analyze(logs, agent_name)
    return {"feedback_summary": summary}
