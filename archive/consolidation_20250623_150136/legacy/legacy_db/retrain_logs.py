import json
import os
import datetime
from typing import List, Dict

LOG_FILE = os.path.join(os.path.dirname(__file__), "retrain_logs.json")

from src.legacy_config.models import RetrainLog
from src.legacy_db.session import get_session

def append_retrain_log(agent: str, status: str, output: str):
    with get_session() as session:
        log = RetrainLog(agent=agent, status=status, output=output)
        session.add(log)
        session.commit()

def get_retrain_logs(agent: str = None):
    with get_session() as session:
        q = session.query(RetrainLog)
        if agent:
            q = q.filter_by(agent=agent)
        return [log.__dict__ for log in q.order_by(RetrainLog.timestamp.desc()).all()]
