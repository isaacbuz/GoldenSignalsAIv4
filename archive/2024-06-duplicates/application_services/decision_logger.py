from datetime import datetime
import json

class DecisionLogger:
    def __init__(self):
        self.log_buffer = []
        self.indicators_to_show = ['MA_Confluence', 'RSI', 'MACD_Strength', 'VWAP_Score', 'Volume_Spike']

    async def log_decision_process(self, symbol, decision):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': decision.action.name,
            'confidence': decision.confidence,
            'entry_price': decision.entry_price,
            'stop_loss': decision.stop_loss,
            'take_profit': decision.take_profit,
            'timeframe': decision.timeframe,
            'rationale': {k: float(v) for k, v in decision.rationale.items()}
        }
        self.log_buffer.append(entry)
        self.log_buffer = self.log_buffer[-100:]
        with open(os.path.join(BASE_DIR, "decision_log.json"), 'w') as f:
            json.dump(self.log_buffer, f, indent=2)

    def get_decision_log(self):
        try:
            with open(os.path.join(BASE_DIR, "decision_log.json"), 'r') as f:
                return json.load(f)
        except:
            return self.log_buffer
