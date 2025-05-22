from fastapi import APIRouter, Request
from application.services.meta_signal_agent import MetaSignalAgent

router = APIRouter()
meta_agent = MetaSignalAgent()

@router.post("/meta_signal/predict")
async def meta_signal_predict(request: Request):
    data = await request.json()
    # Expecting: {"technical": {"signal": ..., "confidence": ...}, ...}
    result = meta_agent.predict(data)
    return result
