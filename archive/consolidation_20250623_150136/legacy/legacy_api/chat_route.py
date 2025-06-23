from fastapi import APIRouter, Request
from agents.meta.signal_summarizer_agent import SignalSummarizerAgent

router = APIRouter()

@router.post("/api/chat")
async def chat_llm(request: Request):
    payload = await request.json()
    symbol = payload.get("symbol", "AAPL")
    agents = payload.get("agent_outputs", {})
    meta = payload.get("meta_signal", {})
    use_grok = payload.get("use_grok", False)
    summarizer = SignalSummarizerAgent(use_grok=use_grok)
    explanation = summarizer.summarize(symbol, agents, meta)
    return {"summary": explanation}
