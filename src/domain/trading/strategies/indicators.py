import numpy as np
import pandas as pd


class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def moving_average(self, window):
        return self.data['close'].rolling(window=window).mean()

    def exponential_moving_average(self, window):
        return self.data['close'].ewm(span=window, adjust=False).mean()

    def vwap(self):
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        vwap = (typical_price * self.data['volume']).cumsum() / self.data['volume'].cumsum()
        return vwap

    def bollinger_bands(self, window):
        sma = self.moving_average(window)
        std = self.data['close'].rolling(window=window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        return upper_band, sma, lower_band

    def rsi(self, window):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd(self, fast, slow, signal):
        ema_fast = self.exponential_moving_average(fast)
        ema_slow = self.exponential_moving_average(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
