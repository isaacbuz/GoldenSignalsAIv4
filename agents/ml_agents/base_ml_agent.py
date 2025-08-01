"""
Base ML Agent Framework for GoldenSignalsAI
Foundation for all ML model agents with microservice capabilities
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aioredis
import httpx
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

# Metrics
prediction_counter = Counter('ml_agent_predictions_total', 'Total predictions made', ['agent_type', 'model'])
prediction_latency = Histogram('ml_agent_prediction_duration_seconds', 'Prediction latency', ['agent_type'])
agent_accuracy = Gauge('ml_agent_accuracy', 'Agent accuracy score', ['agent_type'])


class AgentStatus(Enum):
    IDLE = "idle"
    TRAINING = "training"
    PREDICTING = "predicting"
    ERROR = "error"
    UPDATING = "updating"


class MessageType(Enum):
    PREDICTION_REQUEST = "prediction_request"
    TRAINING_REQUEST = "training_request"
    MODEL_UPDATE = "model_update"
    PERFORMANCE_REPORT = "performance_report"
    COLLABORATION_REQUEST = "collaboration_request"


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)


@dataclass
class ModelPrediction:
    """Standardized prediction output"""
    value: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    agent_id: str
    model_version: str


class BaseMLAgent(ABC):
    """Base class for all ML model agents"""

    def __init__(self,
                 agent_id: str,
                 model_type: str,
                 config: Dict[str, Any]):
        self.agent_id = agent_id
        self.model_type = model_type
        self.config = config
        self.status = AgentStatus.IDLE
        self.model = None
        self.model_version = "1.0.0"
        self.performance_history = []
        self.logger = logging.getLogger(f"Agent.{agent_id}")

        # Communication
        self.redis_client = None
        self.http_client = httpx.AsyncClient()
        self.message_queue = asyncio.Queue()
        self.subscribers = set()

        # Performance tracking
        self.predictions_made = 0
        self.last_accuracy = 0.0
        self.last_training_time = None

    async def initialize(self):
        """Initialize agent connections and model"""
        self.redis_client = await aioredis.create_redis_pool('redis://localhost')
        await self.load_model()
        await self.subscribe_to_channels()
        self.logger.info(f"Agent {self.agent_id} initialized")

    async def subscribe_to_channels(self):
        """Subscribe to relevant Redis channels"""
        channels = [
            f"agent:{self.agent_id}",
            f"model:{self.model_type}",
            "broadcast:all_agents"
        ]

        for channel in channels:
            await self.redis_client.subscribe(channel)

    @abstractmethod
    async def load_model(self):
        """Load or initialize the ML model"""
        pass

    @abstractmethod
    async def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the model with new data"""
        pass

    @abstractmethod
    async def predict(self, features: pd.DataFrame, **kwargs) -> ModelPrediction:
        """Make predictions using the model"""
        pass

    @abstractmethod
    async def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        pass

    async def handle_message(self, message: AgentMessage):
        """Process incoming messages"""
        self.logger.debug(f"Handling message: {message.message_type}")

        if message.message_type == MessageType.PREDICTION_REQUEST:
            await self.handle_prediction_request(message)
        elif message.message_type == MessageType.TRAINING_REQUEST:
            await self.handle_training_request(message)
        elif message.message_type == MessageType.COLLABORATION_REQUEST:
            await self.handle_collaboration_request(message)
        elif message.message_type == MessageType.MODEL_UPDATE:
            await self.handle_model_update(message)

    async def handle_prediction_request(self, message: AgentMessage):
        """Handle prediction requests"""
        try:
            self.status = AgentStatus.PREDICTING
            features = pd.DataFrame(message.payload['features'])

            with prediction_latency.labels(agent_type=self.model_type).time():
                prediction = await self.predict(features, **message.payload.get('params', {}))

            prediction_counter.labels(
                agent_type=self.model_type,
                model=self.model_version
            ).inc()

            # Send response
            response = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type=MessageType.PREDICTION_REQUEST,
                payload={
                    'prediction': asdict(prediction),
                    'request_id': message.id
                },
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            self.status = AgentStatus.ERROR
        finally:
            self.status = AgentStatus.IDLE

    async def handle_training_request(self, message: AgentMessage):
        """Handle training requests"""
        try:
            self.status = AgentStatus.TRAINING
            data = pd.DataFrame(message.payload['data'])

            results = await self.train(data, **message.payload.get('params', {}))

            self.last_training_time = datetime.now()
            self.model_version = f"{self.model_version.split('.')[0]}.{int(self.model_version.split('.')[1]) + 1}.0"

            # Notify other agents of model update
            await self.broadcast_model_update(results)

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.status = AgentStatus.ERROR
        finally:
            self.status = AgentStatus.IDLE

    async def handle_collaboration_request(self, message: AgentMessage):
        """Handle collaboration requests from other agents"""
        collaboration_type = message.payload.get('type')

        if collaboration_type == 'ensemble':
            await self.participate_in_ensemble(message)
        elif collaboration_type == 'feature_sharing':
            await self.share_features(message)
        elif collaboration_type == 'model_averaging':
            await self.contribute_to_averaging(message)

    async def send_message(self, message: AgentMessage):
        """Send message to another agent or broadcast"""
        channel = f"agent:{message.recipient}"
        await self.redis_client.publish(channel, message.to_json())

    async def broadcast_model_update(self, results: Dict[str, Any]):
        """Broadcast model update to interested agents"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient="all",
            message_type=MessageType.MODEL_UPDATE,
            payload={
                'model_type': self.model_type,
                'version': self.model_version,
                'performance': results,
                'timestamp': datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )

        await self.redis_client.publish("broadcast:model_updates", message.to_json())

    async def collaborate_with_agents(self, agent_ids: List[str], task: str) -> Dict[str, Any]:
        """Collaborate with other agents on a task"""
        collaboration_id = str(uuid.uuid4())
        responses = {}

        for agent_id in agent_ids:
            message = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=agent_id,
                message_type=MessageType.COLLABORATION_REQUEST,
                payload={
                    'task': task,
                    'collaboration_id': collaboration_id
                },
                timestamp=datetime.now(),
                correlation_id=collaboration_id
            )

            await self.send_message(message)

        # Wait for responses (with timeout)
        # Implementation depends on specific collaboration needs

        return responses

    async def run(self):
        """Main agent loop"""
        await self.initialize()

        try:
            while True:
                # Check for messages
                channel, message_data = await self.redis_client.get_message()
                if message_data:
                    message = AgentMessage.from_json(message_data)
                    await self.handle_message(message)

                # Periodic tasks
                await self.perform_maintenance()

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            await self.shutdown()

    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Update metrics
        agent_accuracy.labels(agent_type=self.model_type).set(self.last_accuracy)

        # Check if retraining is needed
        if self.should_retrain():
            await self.request_retraining()

    def should_retrain(self) -> bool:
        """Determine if model needs retraining"""
        if not self.last_training_time:
            return True

        # Retrain if accuracy drops or time threshold exceeded
        time_since_training = (datetime.now() - self.last_training_time).days
        return (self.last_accuracy < 0.7) or (time_since_training > 7)

    async def shutdown(self):
        """Clean shutdown"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        await self.redis_client.close()
        await self.http_client.aclose()


class MLAgentFactory:
    """Factory for creating ML agents"""

    @staticmethod
    def create_agent(agent_type: str, config: Dict[str, Any]) -> BaseMLAgent:
        """Create an ML agent of specified type"""
        from .arima_garch_agent import ARIMAGARCHAgent
        from .lstm_agent import LSTMAgent
        from .prophet_agent import ProphetAgent
        from .transformer_agent import TransformerAgent
        from .xgboost_agent import XGBoostAgent

        agents = {
            'arima_garch': ARIMAGARCHAgent,
            'lstm': LSTMAgent,
            'xgboost': XGBoostAgent,
            'transformer': TransformerAgent,
            'prophet': ProphetAgent
        }

        if agent_type not in agents:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        return agents[agent_type](agent_id, config)
