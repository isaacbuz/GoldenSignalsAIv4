import os
import json
import mlflow
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from infrastructure.config_manager import config_manager
from infrastructure.monitoring import system_monitoring

class ModelRegistry:
    """
    Advanced Machine Learning Model Management System
    
    Features:
    - Model versioning
    - Automated model tracking
    - Performance monitoring
    - Model comparison
    - Experiment management
    """
    
    def __init__(self):
        # MLflow configuration
        mlflow_tracking_uri = config_manager.get('ml.mlflow.tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Model storage configuration
        self.model_base_path = config_manager.get(
            'ml.model_storage_path', 
            os.path.join(os.path.dirname(__file__), 'models')
        )
        os.makedirs(self.model_base_path, exist_ok=True)
    
    @system_monitoring.trace_function
    def register_model(
        self, 
        model: torch.nn.Module, 
        model_name: str, 
        performance_metrics: Dict[str, float]
    ) -> str:
        """
        Register a trained machine learning model
        
        Args:
            model (torch.nn.Module): Trained PyTorch model
            model_name (str): Name of the model
            performance_metrics (Dict[str, float]): Model performance metrics
        
        Returns:
            str: Model version identifier
        """
        with mlflow.start_run():
            # Log model parameters
            mlflow.log_params({
                'model_architecture': model.__class__.__name__,
                'total_parameters': sum(p.numel() for p in model.parameters())
            })
            
            # Log performance metrics
            mlflow.log_metrics(performance_metrics)
            
            # Save model
            model_path = os.path.join(self.model_base_path, f"{model_name}_v{self._get_next_version(model_name)}")
            torch.save(model.state_dict(), model_path)
            
            # Register model in MLflow
            mlflow.pytorch.log_model(model, model_name)
            
            # Create model metadata
            metadata = {
                'name': model_name,
                'path': model_path,
                'performance': performance_metrics,
                'timestamp': str(datetime.now())
            }
            
            with open(f"{model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            return model_path
    
    def load_best_model(
        self, 
        model_name: str, 
        metric: str = 'accuracy'
    ) -> Optional[torch.nn.Module]:
        """
        Load the best performing model for a given model name
        
        Args:
            model_name (str): Name of the model
            metric (str): Performance metric to compare
        
        Returns:
            Optional[torch.nn.Module]: Best performing model
        """
        model_candidates = self._find_model_versions(model_name)
        
        if not model_candidates:
            return None
        
        # Sort models by performance metric
        best_model_path = max(
            model_candidates, 
            key=lambda path: self._load_model_metadata(path).get('performance', {}).get(metric, 0)
        )
        
        return self._load_model(best_model_path)
    
    def _get_next_version(self, model_name: str) -> int:
        """
        Generate next model version number
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            int: Next version number
        """
        existing_versions = self._find_model_versions(model_name)
        return len(existing_versions) + 1
    
    def _find_model_versions(self, model_name: str) -> List[str]:
        """
        Find all versions of a specific model
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            List[str]: Paths to model versions
        """
        return [
            os.path.join(self.model_base_path, f)
            for f in os.listdir(self.model_base_path)
            if f.startswith(model_name) and f.endswith('.pth')
        ]
    
    def _load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Load model metadata
        
        Args:
            model_path (str): Path to model file
        
        Returns:
            Dict[str, Any]: Model metadata
        """
        metadata_path = f"{model_path}_metadata.json"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return {}
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model
        
        Args:
            model_path (str): Path to model file
        
        Returns:
            torch.nn.Module: Loaded model
        """
        model_metadata = self._load_model_metadata(model_path)
        model_class = getattr(
            __import__('models'), 
            model_metadata.get('model_architecture', 'DefaultModel')
        )
        
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model

# Singleton instance
model_registry = ModelRegistry()
