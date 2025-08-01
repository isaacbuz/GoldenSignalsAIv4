"""
ml_registry.py

Advanced Machine Learning Model Management System for GoldenSignalsAI.
Migrated from /ml_infrastructure for unified access by agents and research modules.

Features:
- Model versioning
- Automated model tracking
- Performance monitoring
- Model comparison
- Experiment management
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import torch

from src.infrastructure.config_manager import config_manager
from src.infrastructure.monitoring import system_monitoring


class ModelRegistry:
    """
    Advanced Machine Learning Model Management System
    """
    def __init__(self):
        mlflow_tracking_uri = config_manager.get('ml.mlflow.tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.model_base_path = config_manager.get(
            'ml.model_storage_path',
            os.path.join(os.path.dirname(__file__), 'models')
        )
        os.makedirs(self.model_base_path, exist_ok=True)
    @system_monitoring.trace_function
    def register_model(self, model: torch.nn.Module, model_name: str, performance_metrics: Dict[str, float]) -> str:
        """Register a trained machine learning model."""
        # ... (rest of the registration logic)
        pass
    # ... (rest of the class logic)
