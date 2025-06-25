import numpy as np

class ModelFactory:
    """
    Factory and registry for all supported ML models. Allows runtime selection and config-driven instantiation.
    """
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_model(cls, name: str, config: dict = None, **kwargs) -> 'BaseModel':
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered. Available: {list(cls._registry.keys())}")
        model_cls = cls._registry[name]
        # Merge config and kwargs
        params = dict(config or {})
        params.update(kwargs)
        try:
            return model_cls(**params)
        except Exception as e:
            from src.infrastructure.error_handler import ModelInferenceError, ErrorHandler
            ErrorHandler.handle_error(ModelInferenceError(f"Failed to instantiate model '{name}': {e}"))
            raise

    @classmethod
    def available_models(cls):
        return list(cls._registry.keys())

class BaseModel:
    """
    Abstract base class for all ML models in GoldenSignalsAI.
    """
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


class MockLSTMModel(BaseModel):
    def __init__(self, **kwargs):
        pass
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

class MockXGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        pass
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

class MockLightGBMModel(BaseModel):
    def __init__(self, **kwargs):
        pass
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

class MockSentimentModel(BaseModel):
    def analyze(self, news_articles):
        return 0.1

class MockRLModel(BaseModel):
    def train(self, df):
        pass

# Register models
ModelFactory.register('lstm')(MockLSTMModel)
ModelFactory.register('xgboost')(MockXGBoostModel)
ModelFactory.register('lightgbm')(MockLightGBMModel)
ModelFactory.register('sentiment')(MockSentimentModel)
ModelFactory.register('rl')(MockRLModel)
