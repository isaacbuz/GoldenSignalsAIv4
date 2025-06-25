"""
LSTM-based forecasting agent with robust error handling and validation.
"""
import logging
from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from agents.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class LSTMForecastModel(nn.Module):
    """LSTM forecasting model."""
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
        super(LSTMForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

class LSTMForecastAgent(BaseAgent):
    """Agent for LSTM-based time series forecasting with robust error handling."""
    def __init__(self, lookback: int = 20, model_path: Optional[str] = None):
        super().__init__(name="LSTMForecast", agent_type="forecasting")
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.model = LSTMForecastModel()
        try:
            if model_path:
                self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize/load LSTM model: {e}")
            self.model = None

    def prepare_data(self, series: pd.Series) -> Optional[torch.Tensor]:
        """Prepare time series data for LSTM model."""
        if len(series) <= self.lookback:
            logger.warning(f"Insufficient data: series length {len(series)}, lookback {self.lookback}")
            return None
        data = self.scaler.fit_transform(series.values.reshape(-1, 1))
        sequences = [data[i:i+self.lookback] for i in range(len(data) - self.lookback)]
        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process time series data to generate forecasts."""
        series = pd.Series(data.get("prices", []))
        
        if self.model is None:
            logger.error("LSTM model not initialized.")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "LSTM model not initialized."}
            }
            
        if not isinstance(series, pd.Series) or len(series) <= self.lookback:
            logger.warning("Input series too short or not a pandas Series.")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": f"Series too short for lookback={self.lookback}."}
            }
            
        try:
            X = self.prepare_data(series)
            if X is None or len(X) == 0:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data for prediction."}
                }
                
            with torch.no_grad():
                output = self.model(X[-1].unsqueeze(0))
            prediction = float(self.scaler.inverse_transform(output.numpy())[0][0])
            
            # Generate trading signal based on prediction
            current_price = series.iloc[-1]
            price_change = (prediction - current_price) / current_price
            
            if price_change > 0.01:  # 1% threshold for buy
                action = "buy"
                confidence = min(price_change * 5, 1.0)  # Scale confidence
            elif price_change < -0.01:  # -1% threshold for sell
                action = "sell"
                confidence = min(abs(price_change) * 5, 1.0)
            else:
                action = "hold"
                confidence = 0.0
                
            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "prediction": prediction,
                    "current_price": current_price,
                    "price_change": price_change,
                    "lookback": self.lookback
                }
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
