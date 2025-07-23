import json
from pathlib import Path
from typing import Dict, List


class WatchlistManager:
    def __init__(self, storage_path: str = "data/watchlists.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text(json.dumps({}))

    def _load(self) -> Dict[str, List[Dict]]:
        return json.loads(self.storage_path.read_text())

    def _save(self, data: Dict[str, List[Dict]]):
        self.storage_path.write_text(json.dumps(data, indent=2))

    def add_ticker(self, user_id: str, ticker: str, tags: List[str]):
        data = self._load()
        user_watchlist = data.get(user_id, [])
        user_watchlist.append({"ticker": ticker, "tags": tags})
        data[user_id] = user_watchlist
        self._save(data)

    def get_watchlist(self, user_id: str) -> List[Dict]:
        data = self._load()
        return data.get(user_id, [])

    def filter_by_tag(self, user_id: str, tag: str) -> List[str]:
        watchlist = self.get_watchlist(user_id)
        return [entry["ticker"] for entry in watchlist if tag in entry["tags"]]
