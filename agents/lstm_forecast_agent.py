import logging
from typing import Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

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

class LSTMForecastAgent:
    """Agent for LSTM-based time series forecasting with robust error handling."""
    def __init__(self, lookback: int = 20, model_path: Optional[str] = None):
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
        if len(series) <= self.lookback:
            logger.warning(f"Insufficient data: series length {len(series)}, lookback {self.lookback}")
            return None
        data = self.scaler.fit_transform(series.values.reshape(-1, 1))
        sequences = [data[i:i+self.lookback] for i in range(len(data) - self.lookback)]
        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def predict(self, series: pd.Series) -> Any:
        """Predict the next value in a time series. Returns float prediction or error dict."""
        if self.model is None:
            logger.error("LSTM model not initialized.")
            return {"error": "LSTM model not initialized."}
        if not isinstance(series, pd.Series) or len(series) <= self.lookback:
            logger.warning("Input series too short or not a pandas Series.")
            return {"error": f"Series too short for lookback={self.lookback}."}
        try:
            X = self.prepare_data(series)
            if X is None or len(X) == 0:
                return {"error": "Insufficient data for prediction."}
            with torch.no_grad():
                output = self.model(X[-1].unsqueeze(0))
            prediction = self.scaler.inverse_transform(output.numpy())
            return float(prediction[0][0])
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {"error": str(e)}
