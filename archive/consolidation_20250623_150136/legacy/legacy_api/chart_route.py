import base64
from fastapi import APIRouter, File, UploadFile
from agents.vision.chart_analysis import analyze_chart_image

router = APIRouter()

@router.post("/api/chart/analyze")
async def chart_analyzer(file: UploadFile = File(...)):
    result = analyze_chart_image(file)
    if "error" in result:
        return result
    overlay_b64 = base64.b64encode(result["overlay_image"]).decode()
    return {
        "patterns": result["patterns"],
        "overlay_image": overlay_b64
    }
