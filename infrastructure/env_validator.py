import os
import sys
from typing import Dict, Any, List
from infrastructure.config_manager import config_manager
import logging

class EnvironmentValidator:
    """
    Comprehensive environment configuration validator for GoldenSignalsAI.
    
    Validates:
    - Required environment variables
    - API key configurations
    - Deployment prerequisites
    - System resource requirements
    """
    
    def __init__(self):
        """
        Initialize environment validator with logging and configuration.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
    
    def validate_environment(self) -> bool:
        """
        Perform comprehensive environment validation.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        checks = [
            self._validate_required_env_vars(),
            self._validate_api_configurations(),
            self._validate_system_resources(),
            self._validate_deployment_prerequisites()
        ]
        
        return all(checks)
    
    def _validate_required_env_vars(self) -> bool:
        """
        Validate critical environment variables, including all required API keys.
        
        Returns:
            bool: True if all required variables are set
        """
        required_vars = [
            'APP_ENV',
            'ALPHA_VANTAGE_API_KEY',
            'TWITTER_API_KEY',
            'NEWS_API_KEY',
            'REDIS_URL',
        ]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        return True
    
    def _validate_api_configurations(self) -> bool:
        """
        Validate API configurations and keys, including presence and format of API keys.
        
        Returns:
            bool: True if API configurations are valid
        """
        api_keys = {
            'ALPHA_VANTAGE_API_KEY': os.environ.get('ALPHA_VANTAGE_API_KEY'),
            'TWITTER_API_KEY': os.environ.get('TWITTER_API_KEY'),
            'NEWS_API_KEY': os.environ.get('NEWS_API_KEY'),
        }
        invalid_keys = [name for name, key in api_keys.items() if not key or key == f'your_{name.lower()}']
        if invalid_keys:
            self.logger.error(f"Missing or invalid API keys: {invalid_keys}")
            return False
        # Additional config checks can be added here
        return True
    
    def _validate_system_resources(self) -> bool:
        """
        Validate system resources against configuration requirements.
        
        Returns:
            bool: True if system meets resource requirements
        """
        import psutil
        
        # Get system resource limits from configuration
        max_memory_mb = self.config.get('performance.memory_limit_mb', 4096)
        max_cpu_percent = self.config.get('performance.cpu_limit_percent', 90)
        
        # Check memory
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        if total_memory < max_memory_mb:
            self.logger.warning(f"Insufficient memory: {total_memory}MB < {max_memory_mb}MB")
            return False
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            self.logger.warning(f"Insufficient CPU cores: {cpu_count} < 4")
            return False
        
        return True
    
    def _validate_deployment_prerequisites(self) -> bool:
        """
        Validate deployment prerequisites and dependencies.
        
        Returns:
            bool: True if all prerequisites are met
        """
        try:
            import poetry.poetry
            import torch
            import numpy
            import pandas
        except ImportError as e:
            self.logger.error(f"Missing critical dependency: {e}")
            return False
        
        # Validate Poetry project configuration
        try:
            project_dir = os.path.dirname(os.path.dirname(__file__))
            poetry_file = os.path.join(project_dir, 'pyproject.toml')
            
            if not os.path.exists(poetry_file):
                self.logger.error("Missing pyproject.toml")
                return False
        except Exception as e:
            self.logger.error(f"Deployment prerequisite validation failed: {e}")
            return False
        
        return True
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive system configuration report.
        
        Returns:
            Dict containing system configuration details
        """
        import platform
        import psutil
        
        return {
            'machine_id': self.config.machine_id,
            'environment': self.config.environment,
            'system': {
                'os': platform.platform(),
                'python_version': platform.python_version(),
                'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
                'cpu_cores': psutil.cpu_count(),
                'cpu_frequency_mhz': psutil.cpu_freq().current
            },
            'configuration': {
                'trading_strategies': self.config.get('trading.strategies', {}),
                'risk_management': self.config.get('trading.risk_management', {}),
                'feature_flags': {
                    flag: self.config.get_feature_flag(flag) 
                    for flag in ['machine_learning.sentiment_analysis', 'advanced_trading_strategies.pairs_trading']
                }
            }
        }
    
    def run_preflight_checks(self) -> bool:
        """
        Run comprehensive preflight checks before deployment.
        
        Returns:
            bool: True if all checks pass
        """
        self.logger.info("Running preflight environment checks...")
        
        if not self.validate_environment():
            self.logger.critical("Environment validation failed. Deployment cannot proceed.")
            return False
        
        # Generate and log system report
        report = self.generate_system_report()
        self.logger.info(f"System Report: {report}")
        
        return True

    def log_api_key_statuses(self):
        """
        Log the status (set/missing) of all critical API keys at startup.
        """
        api_keys = [
            'ALPHA_VANTAGE_API_KEY',
            'TWITTER_API_KEY',
            'NEWS_API_KEY',
        ]
        for key in api_keys:
            value = os.environ.get(key)
            if value and not value.startswith('your_'):
                self.logger.info(f"{key} is set.")
            else:
                self.logger.warning(f"{key} is missing or using a placeholder value.")

# Singleton instance for global access
env_validator = EnvironmentValidator()
