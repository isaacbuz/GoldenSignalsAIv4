class ModelFactory:
    def create_model(self, model_type):
        if model_type == "lstm":
            return MockLSTMModel()
        elif model_type == "xgboost":
            return MockXGBoostModel()
        elif model_type == "lightgbm":
            return MockLightGBMModel()
        elif model_type == "sentiment":
            return MockSentimentModel()
        elif model_type == "rl":
            return MockRLModel()
        raise ValueError(f"Unknown model type: {model_type}")

class MockLSTMModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return 280.0

class MockXGBoostModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return 0.05

class MockLightGBMModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return 0.04

class MockSentimentModel:
    def analyze(self, news_articles):
        return 0.1

class MockRLModel:
    def train(self, df):
        pass
