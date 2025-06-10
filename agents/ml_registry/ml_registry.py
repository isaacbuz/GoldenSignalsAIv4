"""
ML model registry for managing and versioning models.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import joblib
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing ML models and their metadata."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            tracking_uri (str, optional): MLflow tracking URI.
        """
        self.tracking_uri = tracking_uri or "sqlite:///ml_registry.db"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        
    def register_model(
        self,
        model: Any,
        name: str,
        version: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> str:
        """Register a model with metadata and metrics.
        
        Args:
            model: The model object to register.
            name (str): Model name.
            version (str): Model version.
            metadata (Dict[str, Any]): Model metadata.
            metrics (Dict[str, float]): Model performance metrics.
            
        Returns:
            str: Model URI in the registry.
        """
        try:
            # Start MLflow run
            with mlflow.start_run() as run:
                # Log model
                model_path = f"models/{name}/{version}"
                mlflow.sklearn.log_model(model, model_path)
                
                # Log metadata
                for key, value in metadata.items():
                    mlflow.log_param(key, value)
                    
                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                    
                # Register model
                model_uri = f"runs:/{run.info.run_id}/{model_path}"
                registered_model = mlflow.register_model(model_uri, name)
                
                logger.info({
                    "message": f"Registered model {name} version {version}",
                    "model_uri": model_uri
                })
                return model_uri
                
        except Exception as e:
            logger.error({"message": f"Model registration failed: {str(e)}"})
            return ""
            
    def load_model(self, name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load a model from the registry.
        
        Args:
            name (str): Model name.
            version (str, optional): Model version. If None, loads latest.
            
        Returns:
            Any: Loaded model object.
        """
        try:
            if version:
                model_uri = f"models:/{name}/{version}"
            else:
                model_uri = f"models:/{name}/latest"
                
            model = mlflow.sklearn.load_model(model_uri)
            logger.info({
                "message": f"Loaded model {name} version {version or 'latest'}"
            })
            return model
            
        except Exception as e:
            logger.error({"message": f"Model loading failed: {str(e)}"})
            return None
            
    def get_model_info(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model metadata and metrics.
        
        Args:
            name (str): Model name.
            version (str, optional): Model version.
            
        Returns:
            Dict[str, Any]: Model information.
        """
        try:
            model_version = self.client.get_latest_versions(name)[0] if not version else \
                          self.client.get_model_version(name, version)
                          
            run = self.client.get_run(model_version.run_id)
            
            return {
                "name": name,
                "version": model_version.version,
                "creation_timestamp": model_version.creation_timestamp,
                "current_stage": model_version.current_stage,
                "metadata": run.data.params,
                "metrics": run.data.metrics
            }
            
        except Exception as e:
            logger.error({"message": f"Failed to get model info: {str(e)}"})
            return {}
            
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.
        
        Returns:
            List[Dict[str, Any]]: List of model information.
        """
        try:
            models = []
            for rm in self.client.list_registered_models():
                latest_versions = self.client.get_latest_versions(rm.name)
                models.append({
                    "name": rm.name,
                    "latest_version": latest_versions[0].version if latest_versions else None,
                    "creation_timestamp": rm.creation_timestamp,
                    "last_updated_timestamp": rm.last_updated_timestamp
                })
            return models
            
        except Exception as e:
            logger.error({"message": f"Failed to list models: {str(e)}"})
            return []
            
    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a model from the registry.
        
        Args:
            name (str): Model name.
            version (str, optional): Model version. If None, deletes all versions.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if version:
                self.client.delete_model_version(name, version)
            else:
                self.client.delete_registered_model(name)
                
            logger.info({
                "message": f"Deleted model {name} version {version or 'all'}"
            })
            return True
            
        except Exception as e:
            logger.error({"message": f"Model deletion failed: {str(e)}"})
            return False 