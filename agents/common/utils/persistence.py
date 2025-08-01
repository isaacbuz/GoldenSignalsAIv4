"""
Persistence utilities for saving and loading models and agent states.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

import joblib
import torch

logger = logging.getLogger(__name__)

def save_model(model: Any, path: str, format: str = "joblib") -> bool:
    """Save a model to disk.

    Args:
        model (Any): Model object to save.
        path (str): Path to save model.
        format (str): Format to save model ('joblib', 'pickle', or 'torch').

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if format == "joblib":
            joblib.dump(model, path)
        elif format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(model, f)
        elif format == "torch":
            torch.save(model.state_dict(), path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved model to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False

def load_model(path: str, format: str = "joblib", model_class: Optional[Any] = None) -> Optional[Any]:
    """Load a model from disk.

    Args:
        path (str): Path to load model from.
        format (str): Format of saved model ('joblib', 'pickle', or 'torch').
        model_class (Any, optional): Class for PyTorch model initialization.

    Returns:
        Optional[Any]: Loaded model if successful, None otherwise.
    """
    try:
        if format == "joblib":
            model = joblib.load(path)
        elif format == "pickle":
            with open(path, "rb") as f:
                model = pickle.load(f)
        elif format == "torch":
            if model_class is None:
                raise ValueError("model_class required for PyTorch models")
            model = model_class()
            model.load_state_dict(torch.load(path))
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Loaded model from {path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def save_agent_state(state: Dict[str, Any], path: str) -> bool:
    """Save agent state to disk.

    Args:
        state (Dict[str, Any]): Agent state dictionary.
        path (str): Path to save state.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved agent state to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save agent state: {e}")
        return False

def load_agent_state(path: str) -> Optional[Dict[str, Any]]:
    """Load agent state from disk.

    Args:
        path (str): Path to load state from.

    Returns:
        Optional[Dict[str, Any]]: State dictionary if successful, None otherwise.
    """
    try:
        with open(path, "r") as f:
            state = json.load(f)
        logger.info(f"Loaded agent state from {path}")
        return state

    except Exception as e:
        logger.error(f"Failed to load agent state: {e}")
        return None
