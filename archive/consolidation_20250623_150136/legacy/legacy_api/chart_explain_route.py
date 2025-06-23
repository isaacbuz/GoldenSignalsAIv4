from fastapi import APIRouter, Request
from openai import OpenAI
import os

router = APIRouter()

@router.post("/api/chart/explain")
async def chart_explain(request: Request):
    payload = await request.json()
    pattern = payload.get("pattern")
    prompt = f"Explain in plain English what a {pattern} pattern means in technical analysis, and how a trader might act on it."
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a trading educator."},
            {"role": "user", "content": prompt}
        ]
    )
    return {"explanation": response.choices[0].message.content.strip()}
