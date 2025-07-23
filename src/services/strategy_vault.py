import json
from pathlib import Path
from typing import Dict, List


class StrategyVault:
    def __init__(self, vault_path: str = "data/strategy_vault.json"):
        self.vault_path = Path(vault_path)
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.vault_path.exists():
            self.vault_path.write_text(json.dumps([]))

    def _load(self) -> List[Dict]:
        return json.loads(self.vault_path.read_text())

    def _save(self, data: List[Dict]):
        self.vault_path.write_text(json.dumps(data, indent=2))

    def list_strategies(self) -> List[Dict]:
        return self._load()

    def save_strategy(self, name: str, logic: Dict, author: str, tags: List[str] = None):
        data = self._load()
        entry = {
            "name": name,
            "logic": logic,
            "author": author,
            "tags": tags or [],
            "version": 1,
            "timestamp": str(Path().stat().st_ctime)
        }
        data.append(entry)
        self._save(data)

    def import_strategy(self, json_string: str) -> Dict:
        entry = json.loads(json_string)
        self.save_strategy(entry["name"], entry["logic"], entry.get("author", "unknown"), entry.get("tags", []))
        return entry

    def export_strategy(self, name: str) -> str:
        data = self._load()
        for entry in data:
            if entry["name"] == name:
                return json.dumps(entry, indent=2)
        return "{}"
