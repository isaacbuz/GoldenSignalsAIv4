import threading
from typing import List, Dict
from datetime import datetime

class SimLogStore:
    _lock = threading.Lock()
    _log: List[Dict] = []

    @classmethod
    def append(cls, entry: Dict):
        with cls._lock:
            entry = dict(entry)
            entry['timestamp'] = datetime.utcnow().isoformat()
            cls._log.append(entry)

    @classmethod
    def get_all(cls) -> List[Dict]:
        with cls._lock:
            return list(reversed(cls._log))  # newest first

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._log = []
