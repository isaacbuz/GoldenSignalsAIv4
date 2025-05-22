from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict
from infrastructure.auth.jwt_utils import verify_jwt_token
from application.services.audit_logger import AuditLogger
import json

router = APIRouter(prefix="/api/v1/audit_trail", tags=["audit-trail"])

audit_logger = AuditLogger()

class AuditLogRequest(BaseModel):
    user_id: str

class AuditLogResponse(BaseModel):
    logs: List[Dict]

@router.post("/logs", response_model=AuditLogResponse)
async def get_audit_logs(request: AuditLogRequest, user=Depends(verify_jwt_token)):
    try:
        log_path = audit_logger.log_file
        logs = []
        if log_path.exists():
            with log_path.open("r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("actor") == request.user_id:
                            logs.append(entry)
                    except Exception:
                        continue
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
