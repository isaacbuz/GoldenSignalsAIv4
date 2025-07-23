import json
from pathlib import Path
from typing import Dict, List


class CollaborationManager:
    def __init__(self, storage_path: str = "data/shared_dashboards.json"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({}))

    def _load(self) -> Dict[str, List[str]]:
        return json.loads(self.path.read_text())

    def _save(self, data: Dict[str, List[str]]):
        self.path.write_text(json.dumps(data, indent=2))

    def share_dashboard(self, owner_id: str, shared_with_email: str):
        data = self._load()
        if owner_id not in data:
            data[owner_id] = []
        if shared_with_email not in data[owner_id]:
            data[owner_id].append(shared_with_email)
        self._save(data)

    def get_shared_users(self, owner_id: str) -> List[str]:
        data = self._load()
        return data.get(owner_id, [])

    def can_access(self, owner_id: str, user_email: str) -> bool:
        return user_email in self.get_shared_users(owner_id)
