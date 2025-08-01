import json
import logging
import os
import socket
import uuid
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import yaml


class ConfigManager:
    """
    Advanced configuration management system for GoldenSignalsAI.

    Features:
    - Multi-source configuration loading
    - Environment-specific overrides
    - Dynamic feature flags
    - Secure secret management
    - Logging and error handling
    """

    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation.

        Returns:
            ConfigManager: Singleton instance
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_path: Optional[str] = None,
        env: str = 'development'
    ):
        """
        Initialize advanced configuration manager.

        Args:
            config_path (Optional[str]): Path to configuration file
            env (str): Deployment environment
        """
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return

        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Configuration paths
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config.yaml'
        )

        # Environment management
        self.environment = env

        # Configuration storage
        self._config = {}
        self._secrets = {}
        self._feature_flags = {}

        # Machine-specific identifier
        self._machine_id = self._generate_machine_id()

        # Load configurations
        self._load_config()
        self._load_secrets()
        self._load_feature_flags()

        # Mark as initialized
        self._initialized = True

    def _generate_machine_id(self) -> str:
        """
        Generate a unique machine identifier.

        Returns:
            str: Machine-specific unique ID
        """
        return str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{socket.gethostname()}:{os.getpid()}"
        ))

    def _load_config(self):
        """
        Load multi-source configuration with environment-specific overrides.
        """
        try:
            # Load base configuration
            with open(self.config_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}

            # Environment-specific configuration
            env_config = base_config.get(self.environment, {})

            # Merge configurations
            self._config = {**base_config, **env_config}

            # Environment variable overrides
            for key, value in os.environ.items():
                if key.startswith('GOLDENSIGNALS_'):
                    config_key = key[len('GOLDENSIGNALS_'):].lower()
                    self._config[config_key] = self._parse_config_value(value)

            self.logger.info(f"Configuration loaded for {self.environment} environment")

        except Exception as e:
            self.logger.error(f"Configuration loading error: {e}")
            self._config = {}

    def _load_secrets(self):
        """
        Load and manage sensitive configuration secrets.
        Supports multiple secret sources: files, environment variables, secret managers.
        """
        try:
            # Check for secrets file
            secrets_path = os.path.join(
                os.path.dirname(self.config_path),
                'secrets.yaml'
            )

            if os.path.exists(secrets_path):
                with open(secrets_path, 'r') as f:
                    self._secrets = yaml.safe_load(f) or {}

            # Load secrets from environment variables
            for key, value in os.environ.items():
                if key.startswith('SECRET_'):
                    secret_key = key[len('SECRET_'):].lower()
                    self._secrets[secret_key] = value

            self.logger.info("Secrets loaded successfully")

        except Exception as e:
            self.logger.warning(f"Secret loading error: {e}")
            self._secrets = {}

    def _load_feature_flags(self):
        """
        Load and manage dynamic feature flags.
        """
        try:
            flags_path = os.path.join(
                os.path.dirname(self.config_path),
                'feature_flags.json'
            )

            if os.path.exists(flags_path):
                with open(flags_path, 'r') as f:
                    self._feature_flags = json.load(f)

            self.logger.info("Feature flags loaded successfully")

        except Exception as e:
            self.logger.warning(f"Feature flags loading error: {e}")
            self._feature_flags = {}

    def _parse_config_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Parse configuration value with type inference.

        Args:
            value (str): Configuration value to parse

        Returns:
            Union[str, int, float, bool]: Parsed configuration value
        """
        try:
            # Try parsing as JSON to handle complex types
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Fallback to simple type conversion
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value

    @lru_cache(maxsize=None)
    def get(self, key: str, default: Any = None, secret: bool = False) -> Any:
        """
        Retrieve a configuration value with advanced retrieval options.

        Args:
            key (str): Configuration key to retrieve
            default (Any, optional): Default value if key not found
            secret (bool, optional): Retrieve from secrets instead of config

        Returns:
            Any: Configuration or secret value
        """
        source = self._secrets if secret else self._config
        return source.get(key, default)

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """
        Retrieve a feature flag with optional default.

        Args:
            flag_name (str): Name of the feature flag
            default (bool, optional): Default flag state

        Returns:
            bool: Feature flag state
        """
        return self._feature_flags.get(flag_name, default)

    def set_feature_flag(self, flag_name: str, value: bool):
        """
        Dynamically set a feature flag.

        Args:
            flag_name (str): Name of the feature flag
            value (bool): Flag state
        """
        self._feature_flags[flag_name] = value
        self.logger.info(f"Feature flag '{flag_name}' set to {value}")

    def is_feature_enabled(self, flag_name: str) -> bool:
        """
        Return True if the feature flag is enabled, False otherwise.
        """
        return self.get_feature_flag(flag_name, False)

    @property
    def machine_id(self) -> str:
        """
        Retrieve the unique machine identifier.

        Returns:
            str: Machine-specific unique ID
        """
        return self._machine_id

# Singleton instance for global access
config_manager = ConfigManager()
