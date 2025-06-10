from dataclasses import dataclass
from datetime import datetime

@dataclass
class SignalEvent:
    type: str = "SignalEvent"
    symbol: str
    action: str
    price: float
    timestamp: datetime = datetime.now()
