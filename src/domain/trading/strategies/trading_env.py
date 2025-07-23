import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    def __init__(self, data, symbol):
        super(TradingEnv, self).__init__()
        self.data = data
        self.symbol = symbol
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.cash = 10000
        self.shares = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.cash = 10000
        self.shares = 0
        return self._get_observation()

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0
        done = False
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.shares = self.cash // current_price
            self.cash -= self.shares * current_price
        elif action == 2 and self.position == 1:
            self.position = 0
            profit = (current_price - self.entry_price) * self.shares
            self.cash += self.shares * current_price
            self.shares = 0
            reward = profit
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.data.iloc[self.current_step].values
