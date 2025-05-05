import logging
import pandas as pd
import asyncio
from GoldenSignalsAI.domain.trading.strategies.signal_engine import SignalEngine
from GoldenSignalsAI.application.events.event_publisher import EventPublisher
from GoldenSignalsAI.infrastructure.data.fetchers.realtime_fetcher import fetch_realtime_data

logger = logging.getLogger(__name__)

class SignalEngineService:
    def __init__(self):
        self.event_publisher = EventPublisher()

    async def monitor_and_generate_signal(self, symbol):
        while True:
            data = await fetch_realtime_data(symbol)
            if data is None:
                logger.error(f"Failed to fetch data for {symbol}")
                await asyncio.sleep(60)
                continue
            signal_engine = SignalEngine(data)
            signal = signal_engine.generate_signal(symbol)
            logger.info(f"Generated signal for {symbol}: {signal}")
            if signal["action"] in ["Buy", "Sell"]:
                event = {"type": "SignalEvent", "symbol": symbol, "action": signal["action"], "price": signal["price"]}
                await self.event_publisher.publish("signals", event)
            await asyncio.sleep(60)

if __name__ == "__main__":
    service = SignalEngineService()
    asyncio.run(service.monitor_and_generate_signal("TSLA"))
