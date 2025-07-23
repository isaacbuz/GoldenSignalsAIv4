"""
distributed_computing.py
Purpose: Provides distributed computing utilities for GoldenSignalsAI using Ray, enabling parallel model training and prediction across nodes.
"""

import ray


class DistributedComputing:
    def __init__(self):
        ray.init()

    @ray.remote
    def train_model(model, X, y):
        model.fit(X, y)
        return model

    async def parallel_train(self, model, X, y):
        return await self.train_model.remote(model, X, y)

    async def parallel_predict(self, model, X):
        return await model.predict.remote(X)

    def close(self):
        ray.shutdown()
