import mlflow

class MLflowRegistry:
    """
    Handles model tracking, registration, and experiment logging using MLflow.
    """
    def __init__(self, experiment_name="FintechSignals"):
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_params_and_metrics(self, params: dict, metrics: dict):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path: str):
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, artifact_path)
