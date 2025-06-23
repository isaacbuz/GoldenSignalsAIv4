import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockPreprocessor:
    def preprocess(self, df, use_numba=True):
        X = df[['open', 'high', 'low', 'volume']].values
        y = df['close'].shift(-1).values[:-1]
        X = X[:-1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler
