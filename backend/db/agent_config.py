import json
import os
from typing import Dict

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "agent_config_store.json")

from backend.models import AgentConfig
from backend.db.session import get_session

def load_agent_config(agent: str) -> dict:
    with get_session() as session:
        config = session.query(AgentConfig).filter_by(agent=agent).first()
        return config.config if config else {}

def save_agent_config(agent: str, config: dict):
    with get_session() as session:
        obj = session.query(AgentConfig).filter_by(agent=agent).first()
        if obj:
            obj.config = config
        else:
            obj = AgentConfig(agent=agent, config=config)
            session.add(obj)
        session.commit()
