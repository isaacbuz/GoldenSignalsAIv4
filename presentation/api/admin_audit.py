# admin_audit.py
# Audit logging for all sensitive admin actions
from datetime import datetime
import os

AUDIT_LOG_FILE = os.getenv("ADMIN_AUDIT_LOG", "./logs/admin_audit.log")

def log_admin_action(user, action, target=None, outcome="success", details=None):
    line = (
        f"{datetime.utcnow().isoformat()} | "
        f"{user.get('email', 'unknown')} | "
        f"{action} | "
        f"{target or ''} | "
        f"{outcome} | "
        f"{details or ''}\n"
    )
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(line)
