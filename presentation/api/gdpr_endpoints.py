from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/gdpr", tags=["gdpr"])

@router.post("/request-data")
async def request_data(request: Request):
    # TODO: Implement logic to fetch all user data
    return JSONResponse({"message": "Your data request has been received. You will be contacted shortly."})

@router.post("/delete-data")
async def delete_data(request: Request):
    # TODO: Implement logic to delete all user data
    return JSONResponse({"message": "Your data deletion request has been received. Your data will be deleted in accordance with GDPR/CCPA."})
