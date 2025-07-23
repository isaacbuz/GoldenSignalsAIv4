from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    symbol: str
    quantity: int
    price: float
    timestamp: datetime = datetime.now()
    action: str = "Buy"
    stop_loss: float = None
    take_profit: float = None
