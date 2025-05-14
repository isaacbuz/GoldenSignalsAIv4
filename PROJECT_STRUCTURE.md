# GoldenSignalsAI Project Structure

## Directory Overview

### Root Level
- `agents/`: Multi-agent trading signal generation system
- `infrastructure/`: Core system infrastructure
- `services/`: Business logic implementations
- `models/`: Data models and schemas
- `tests/`: Comprehensive test suite
- `utils/`: Utility functions and helpers

### Detailed Component Breakdown

#### Agents
- `predictive/`: Predictive modeling agents
- `sentiment/`: Sentiment analysis agents
- `risk/`: Risk assessment agents

#### Infrastructure
- `data_processing/`: Data ingestion and transformation
- `api_clients/`: External API integrations

#### Services
- `signal_engine/`: Core signal generation logic
- `trading_strategies/`: Trading strategy implementations

## Architectural Principles
- Modular Design
- Microservices Architecture
- Type Safety
- Comprehensive Testing

## Technology Stack
- Language: Python 3.10+
- Framework: FastAPI
- Machine Learning: TensorFlow/PyTorch
- Dependency Management: Poetry
- Containerization: Docker
- CI/CD: GitHub Actions

## Overview
The GoldenSignalsAI project is designed with a modular, scalable architecture that separates concerns and promotes maintainability. This document provides a comprehensive guide to the project's directory structure and the purpose of each component.

## Directory Structure

### Root Level
- `run_services.py`: Main entry point for starting all services
- `pyproject.toml`: Poetry dependency management
- `config.yaml`: Global configuration settings
- `Dockerfile` & `Dockerfile.frontend`: Docker configurations
- `docker-compose.yml`: Multi-service container orchestration

### Core Directories

#### 1. `application/`
Contains core application logic and services.

**Subdirectories**:
- `ai_service/`: AI model orchestration
  - `model_factory.py`: Creates and manages AI models
  - `orchestrator.py`: Coordinates model predictions

- `services/`: Core business logic
  - `signal_engine.py`: Generates trading signals
  - `risk_manager.py`: Manages trading risks
  - `custom_algorithm.py`: Custom trading strategy implementation

- `workflows/`: Agent coordination
  - `agentic_cycle.py`: Implements agent workflow management

#### 2. `domain/`
Defines data models and trading logic.

**Subdirectories**:
- `models/`: Pydantic data validation models
  - `stock.py`: Stock data model
  - `options.py`: Options chain model
  - `signal.py`: Trading signal model

- `trading/`: Trading-specific logic
  - `indicators.py`: Technical indicators
  - `regime_detector.py`: Market regime identification

#### 3. `agents/`
Multi-agent system implementation.

**Subdirectories**:
- `predictive/`: Signal generation agents
  - `breakout.py`: Breakout pattern detection
  - `options_flow.py`: Options flow analysis
  - `regime.py`: Market regime identification

- `sentiment/`: Sentiment analysis agents
  - `news.py`: News sentiment analysis
  - `social_media.py`: Social media sentiment tracking

- `risk/`: Risk management agents
  - `options_risk.py`: Options risk evaluation

#### 4. `infrastructure/`
External data integration.
- `data_fetcher.py`: Fetches market data from external sources

#### 5. `orchestration/`
Manages agent coordination and data streaming.
- `supervisor.py`: Coordinates agent tasks
- `data_feed.py`: Real-time market data streaming

#### 6. `optimization/`
Strategy optimization techniques.
- `genetic.py`: Genetic algorithm for parameter optimization
- `rl_optimizer.py`: Reinforcement learning optimization
- `performance_tracker.py`: Strategy performance metrics

#### 7. `governance/`
Ensures compliance and auditing.
- `constraints.py`: Trading constraint enforcement
- `audit.py`: Action logging and auditing

#### 8. `monitoring/`
Performance monitoring and dashboarding.
- `agent_dashboard.py`: Dash-based performance dashboard

#### 9. `backtesting/`
Strategy backtesting.
- `options_backtest.py`: Options trading strategy backtesting

#### 10. `presentation/`
User interface and API.
- `api/main.py`: FastAPI backend
- `frontend/`: React-based dashboard
- `tests/`: API and component testing

#### 11. `notifications/`
Multi-channel alert system.
- `alert_manager.py`: Manages trade signal notifications

#### 12. `k8s/`
Kubernetes deployment configurations.
- `deployment.yaml`: Service deployment specifications

## Project Generation
The project was generated using a custom script `create_project.py`, which ensures a consistent and comprehensive structure.

## Generating Project Tree
Use the `generate_project_tree.py` script to dynamically generate the project structure:
```bash
python generate_project_tree.py
```

## Best Practices
- Modular design
- Separation of concerns
- Scalable architecture
- Comprehensive testing
- Flexible configuration

## Future Improvements
- Enhanced documentation
- More comprehensive testing
- Real-time API integrations
- Advanced machine learning models

## Contributing
Please refer to `CONTRIBUTING.md` for guidelines on contributing to the project.

## License
This project is licensed under the MIT License. See `LICENSE.md` for details.
