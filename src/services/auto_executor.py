from datetime import datetime, time
import asyncio
from GoldenSignalsAI.application.ai_service.autonomous_engine import Action
from GoldenSignalsAI.infrastructure.external_services.alpaca_trader import AlpacaTrader

class AutoExecutor:
    def __init__(self):
        self.trading_hours = {
            'premarket': (time(4,0), time(9,30)),
            'regular': (time(9,30), time(16,0)),
            'postmarket': (time(16,0), time(20,0))
        }

    def _get_market_phase(self, current_time):
        for phase, (start, end) in self.trading_hours.items():
            if start <= current_time <= end:
                return phase
        return 'closed'

    async def run_intraday_cycle(self, symbols, engine, orchestrator):
        for symbol in symbols:
            data = await orchestrator.data_service.fetch_multi_timeframe_data(symbol)
            if not data:
                continue
            decision = await engine.analyze_and_decide(symbol, data)
            if decision.action != Action.HOLD:
                await self._execute_trade(decision, orchestrator)
                await orchestrator.logger.log_decision_process(symbol, decision)

    async def _execute_trade(self, decision, orchestrator):
        trader = AlpacaTrader()
        qty = 10
        order = trader.place_order(decision.symbol, decision.action.name, qty, decision.entry_price)
        if order and decision.action == Action.LONG and decision.stop_loss:
            trader.set_stop_loss(decision.symbol, qty, decision.stop_loss)
        event = {
            "type": "SignalEvent",
            "symbol": decision.symbol,
            "action": decision.action.name,
            "price": decision.entry_price,
            "confidence_score": decision.confidence
        }
        await orchestrator.alert_service.send_alert(
            user_prefs={"enabled_channels": ["sms", "whatsapp", "x"]}, event=event
        )
