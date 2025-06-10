from fastapi import APIRouter, Request
from agents.feedback_loop import FeedbackLoop

router = APIRouter()
feedback_loop = FeedbackLoop()

@router.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    signal = data.get('signal')
    outcome = data.get('outcome')
    user_rating = data.get('user_rating')
    feedback_loop.add_feedback(signal, outcome, user_rating)
    return {"success": True}

@router.get("/feedback")
def get_feedback():
    return {"feedback": feedback_loop.get_feedback()}
