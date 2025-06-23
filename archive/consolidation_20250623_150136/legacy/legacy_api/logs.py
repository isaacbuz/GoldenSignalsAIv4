from fastapi import APIRouter, Query
import json

router = APIRouter()

@router.get("/api/logs")
def get_logs(min_confidence: int = Query(0), strategy: str = None, agent: str = None):
    try:
        with open("logs/signal_runs.jsonl", "r") as f:
            entries = [json.loads(line) for line in f]

        results = []
        for entry in entries:
            if entry['blended']['confidence'] < min_confidence:
                continue
            if strategy and strategy.lower() not in entry['blended'].get('strategy', '').lower():
                continue
            if agent and agent not in entry['agents']:
                continue
            results.append(entry)

        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": str(e), "results": []}

