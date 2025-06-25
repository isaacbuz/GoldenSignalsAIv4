import os
import subprocess
import logging
from typing import Dict, List, Optional
from src.infrastructure.config_manager import config_manager
from src.infrastructure.env_validator import env_validator

class DeploymentManager:
    """
    Advanced deployment management system for GoldenSignalsAI.
    
    Supports:
    - Docker and Poetry-based deployments
    - Multi-environment configuration
    - Containerization strategies
    - Deployment validation
    """
    
    def __init__(self):
        """
        Initialize deployment manager with logging and configuration.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
    
    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """
        Run a shell command with logging and error handling.
        
        Args:
            command (List[str]): Command to execute
        
        Returns:
            subprocess.CompletedProcess: Command execution result
        """
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )
            self.logger.info(f"Command executed: {' '.join(command)}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            raise
    
    def build_docker_image(
        self, 
        tag: Optional[str] = None, 
        dockerfile_path: Optional[str] = None
    ) -> str:
        """
        Build Docker image for the application.
        
        Args:
            tag (Optional[str]): Custom image tag
            dockerfile_path (Optional[str]): Path to Dockerfile
        
        Returns:
            str: Docker image tag
        """
        # Validate environment before deployment
        if not env_validator.run_preflight_checks():
            raise RuntimeError("Preflight checks failed. Deployment aborted.")
        
        # Determine project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        # Default Dockerfile and tag
        dockerfile_path = dockerfile_path or os.path.join(project_root, 'Dockerfile')
        tag = tag or f"goldensignals-ai:{self.config.environment}"
        
        # Build Docker image
        build_command = [
            'docker', 'build', 
            '-t', tag, 
            '-f', dockerfile_path, 
            project_root
        ]
        
        self._run_command(build_command)
        return tag
    
    def deploy_docker_compose(
        self, 
        compose_file: Optional[str] = None
    ) -> None:
        """
        Deploy application using Docker Compose.
        
        Args:
            compose_file (Optional[str]): Path to docker-compose file
        """
        # Determine project root and compose file
        project_root = os.path.dirname(os.path.dirname(__file__))
        compose_file = compose_file or os.path.join(project_root, 'docker-compose.yml')
        
        # Validate compose file exists
        if not os.path.exists(compose_file):
            raise FileNotFoundError(f"Docker Compose file not found: {compose_file}")
        
        # Deploy with Docker Compose
        deploy_command = [
            'docker-compose', 
            '-f', compose_file, 
            'up', '-d', '--build'
        ]
        
        self._run_command(deploy_command)
    
    def deploy_poetry(self) -> None:
        """
        Deploy application using Poetry for local development.
        """
        # Validate environment
        if not env_validator.run_preflight_checks():
            raise RuntimeError("Preflight checks failed. Deployment aborted.")
        
        # Install dependencies
        install_command = ['poetry', 'install']
        self._run_command(install_command)
        
        # Run database migrations (if applicable)
        migrate_command = ['poetry', 'run', 'alembic', 'upgrade', 'head']
        self._run_command(migrate_command)
        
        # Start application
        start_command = ['poetry', 'run', 'uvicorn', 'main:app', '--reload']
        self._run_command(start_command)
    
    def generate_deployment_report(self) -> Dict:
        """
        Generate comprehensive deployment report.
        
        Returns:
            Dict containing deployment details
        """
        system_report = env_validator.generate_system_report()
        
        return {
            'deployment_mode': self.config.get('app.deployment_mode', 'local'),
            'environment': self.config.environment,
            'system': system_report['system'],
            'active_features': {
                flag: self.config.get_feature_flag(flag)
                for flag in [
                    'machine_learning.sentiment_analysis', 
                    'advanced_trading_strategies.pairs_trading'
                ]
            },
            'risk_management': self.config.get('trading.risk_management', {})
        }

# Singleton instance for global access
deployment_manager = DeploymentManager()
