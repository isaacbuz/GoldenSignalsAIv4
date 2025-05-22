from fastapi import APIRouter, Request
from application.services.gpt_model_copilot import GPTModelCopilot
import os

router = APIRouter()
gpt_agent = GPTModelCopilot(api_key=os.getenv("OPENAI_API_KEY", ""))

@router.post("/gpt_model_copilot/critique")
async def critique_features(request: Request):
    data = await request.json()
    result = gpt_agent.critique_features(data)
    return {"critique": result}
