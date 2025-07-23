from dataclasses import dataclass
from datetime import datetime


@dataclass
class PriceAlertEvent:
    type: str = "PriceAlertEvent"
    symbol: str
    threshold: float
    price: float
    timestamp: datetime = datetime.now()
