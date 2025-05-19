import logging
from datetime import datetime
from infrastructure.storage.s3_storage import S3Storage

class AuditLogger:
    def __init__(self):
        self.s3_storage = S3Storage()
        self.logger = logging.getLogger(__name__)

    def log_event(self, event_type, details):
        timestamp = datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "event_type": event_type, "details": details}
        self.s3_storage.save_log(log_entry)
        self.logger.info(f"Audit log: {log_entry}")
