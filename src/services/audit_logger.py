import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    def __init__(self, log_path: str = "logs/admin_audit.log"):
        self.log_file = Path(log_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, actor: str, action: str, status: str = "success", metadata: dict = None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "status": status,
            "metadata": metadata or {},
        }
        with self.log_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
