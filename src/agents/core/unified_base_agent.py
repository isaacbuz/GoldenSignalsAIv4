"""
Unified Base Agent Framework for GoldenSignalsAI
Core foundation for all agent types with microservices capabilities
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
import numpy as np
import pandas as pd

# External dependencies
import aioredis
import httpx
from prometheus_client import Counter, Histogram, Gauge, Summary
import grpc
from fastapi import FastAPI, WebSocket
import ray

# Metrics
agent_messages_total = Counter('agent_messages_total', 'Total messages processed', ['agent_type', 'message_type'])
agent_processing_time = Histogram('agent_processing_duration_seconds', 'Processing time', ['agent_type', 'operation'])
agent_health_status = Gauge('agent_health_status', 'Agent health (1=healthy, 0=unhealthy)', ['agent_id'])
agent_performance_score = Gauge('agent_performance_score', 'Agent performance score', ['agent_id', 'metric'])


class AgentType(Enum):
    """Types of agents in the system"""
    ML_MODEL = "ml_model"
    STRATEGY = "strategy"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    DATA = "data"
    ORCHESTRATOR = "orchestrator"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class AgentMessage:
    """Unified message format for all agent communication"""
    id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    avg_processing_time: float = 0.0
    success_rate: float = 1.0


class UnifiedBaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: AgentType,
                 config: Dict[str, Any]):
        # Identity
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.name = config.get('name', agent_id)
        self.version = config.get('version', '1.0.0')
        
        # State
        self.is_running = False
        self.is_healthy = True
        self.last_heartbeat = datetime.now()
        
        # Capabilities
        self.capabilities: Dict[str, AgentCapability] = {}
        self._register_capabilities()
        
        # Communication
        self.redis_client = None
        self.grpc_channel = None
        self.http_client = httpx.AsyncClient()
        self.websocket_connections: List[WebSocket] = []
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._register_message_handlers()
        
        # Performance tracking
        self.messages_processed = 0
        self.errors_count = 0
        self.performance_metrics: Dict[str, float] = {}
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")
        
        # Subscriptions
        self.subscribed_topics: List[str] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    def _register_capabilities(self):
        """Register agent capabilities"""
        pass
    
    @abstractmethod
    def _register_message_handlers(self):
        """Register message type handlers"""
        pass
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific request based on agent's specialization"""
        pass
    
    async def initialize(self):
        """Initialize agent connections and resources"""
        try:
            # Redis connection
            self.redis_client = await aioredis.create_redis_pool(
                self.config.get('redis_url', 'redis://localhost')
            )
            
            # Subscribe to agent channels
            await self._subscribe_to_channels()
            
            # Register with service discovery
            await self._register_agent()
            
            # Start health check
            asyncio.create_task(self._health_check_loop())
            
            self.is_running = True
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def _subscribe_to_channels(self):
        """Subscribe to relevant Redis channels"""
        channels = [
            f"agent:{self.agent_id}",  # Direct messages
            f"agent_type:{self.agent_type.value}",  # Type-specific broadcasts
            "agent:broadcast",  # System-wide broadcasts
        ]
        
        # Add custom subscriptions
        channels.extend(self.config.get('subscribe_to', []))
        
        for channel in channels:
            await self.redis_client.subscribe(channel)
            self.subscribed_topics.append(channel)
    
    async def _register_agent(self):
        """Register agent with service discovery"""
        registration = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'name': self.name,
            'version': self.version,
            'capabilities': [cap.name for cap in self.capabilities.values()],
            'endpoint': f"grpc://localhost:{self.config.get('grpc_port', 50051)}",
            'health_check': f"http://localhost:{self.config.get('http_port', 8000)}/health",
            'registered_at': datetime.now().isoformat()
        }
        
        await self.redis_client.setex(
            f"agent:registry:{self.agent_id}",
            86400,  # 24 hour TTL
            json.dumps(registration)
        )
    
    async def send_message(self, 
                          recipient_id: str,
                          message_type: str,
                          payload: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL,
                          wait_for_reply: bool = False,
                          timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Send message to another agent"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Track metrics
        agent_messages_total.labels(
            agent_type=self.agent_type.value,
            message_type=message_type
        ).inc()
        
        if wait_for_reply:
            # Create future for response
            future = asyncio.Future()
            self.pending_requests[message.id] = future
            
            # Send message
            await self._publish_message(message)
            
            try:
                # Wait for response
                response = await asyncio.wait_for(future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"Request {message.id} timed out")
                del self.pending_requests[message.id]
                return None
        else:
            # Fire and forget
            await self._publish_message(message)
            return None
    
    async def _publish_message(self, message: AgentMessage):
        """Publish message to Redis"""
        channel = f"agent:{message.recipient_id}"
        await self.redis_client.publish(channel, message.to_json())
    
    async def broadcast(self, 
                       message_type: str,
                       payload: Dict[str, Any],
                       target_type: Optional[AgentType] = None):
        """Broadcast message to multiple agents"""
        channel = f"agent_type:{target_type.value}" if target_type else "agent:broadcast"
        
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id="*",  # Broadcast
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now()
        )
        
        await self.redis_client.publish(channel, message.to_json())
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        with agent_processing_time.labels(
            agent_type=self.agent_type.value,
            operation='handle_message'
        ).time():
            try:
                # Check if this is a reply to a pending request
                if message.reply_to and message.reply_to in self.pending_requests:
                    future = self.pending_requests.pop(message.reply_to)
                    future.set_result(message.payload)
                    return
                
                # Route to appropriate handler
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    response = await handler(message)
                    
                    # Send reply if needed
                    if response is not None:
                        await self.send_message(
                            recipient_id=message.sender_id,
                            message_type=f"{message.message_type}_response",
                            payload=response,
                            priority=message.priority
                        )
                else:
                    self.logger.warning(f"No handler for message type: {message.message_type}")
                
                self.messages_processed += 1
                
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
                self.errors_count += 1
    
    async def collaborate(self, 
                         agent_ids: List[str],
                         task: str,
                         params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collaborate with multiple agents on a task"""
        collaboration_id = str(uuid.uuid4())
        
        # Send collaboration requests
        futures = []
        for agent_id in agent_ids:
            future = self.send_message(
                recipient_id=agent_id,
                message_type="collaboration_request",
                payload={
                    'task': task,
                    'params': params,
                    'collaboration_id': collaboration_id
                },
                wait_for_reply=True
            )
            futures.append(future)
        
        # Wait for all responses
        responses = await asyncio.gather(*futures, return_exceptions=True)
        
        # Filter out errors
        valid_responses = [r for r in responses if isinstance(r, dict)]
        
        return valid_responses
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event that other agents can subscribe to"""
        event = {
            'event_type': event_type,
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        await self.redis_client.publish(f"event:{event_type}", json.dumps(event))
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to events from other agents"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _health_check_loop(self):
        """Periodic health check"""
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check agent health
                health_status = await self.check_health()
                self.is_healthy = health_status['healthy']
                
                # Update metrics
                agent_health_status.labels(agent_id=self.agent_id).set(
                    1 if self.is_healthy else 0
                )
                
                # Update performance metrics
                for metric, value in self.performance_metrics.items():
                    agent_performance_score.labels(
                        agent_id=self.agent_id,
                        metric=metric
                    ).set(value)
                
                # Update registry
                await self._update_registry()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    async def check_health(self) -> Dict[str, Any]:
        """Check agent health - override for custom checks"""
        return {
            'healthy': True,
            'agent_id': self.agent_id,
            'uptime': (datetime.now() - self.last_heartbeat).total_seconds(),
            'messages_processed': self.messages_processed,
            'error_rate': self.errors_count / max(self.messages_processed, 1)
        }
    
    async def _update_registry(self):
        """Update agent information in registry"""
        info = {
            'last_seen': datetime.now().isoformat(),
            'healthy': self.is_healthy,
            'messages_processed': self.messages_processed,
            'performance': self.performance_metrics
        }
        
        await self.redis_client.hset(
            f"agent:status:{self.agent_id}",
            mapping=info
        )
    
    async def run(self):
        """Main agent loop"""
        await self.initialize()
        
        # Start message processing
        message_task = asyncio.create_task(self._process_messages())
        
        # Start Redis subscription handler
        subscription_task = asyncio.create_task(self._handle_subscriptions())
        
        try:
            await asyncio.gather(message_task, subscription_task)
        except asyncio.CancelledError:
            await self.shutdown()
    
    async def _process_messages(self):
        """Process messages from queue"""
        while self.is_running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                await self.handle_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    async def _handle_subscriptions(self):
        """Handle Redis subscriptions"""
        while self.is_running:
            try:
                # Get message from Redis
                message = await self.redis_client.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                
                if message:
                    # Parse and queue message
                    agent_message = AgentMessage.from_json(message['data'])
                    await self.message_queue.put(agent_message)
                    
            except Exception as e:
                self.logger.error(f"Subscription error: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.is_running = False
        
        # Close connections
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        await self.http_client.aclose()
        
        # Deregister from service discovery
        await self.redis_client.delete(f"agent:registry:{self.agent_id}")
    
    # Convenience methods for specific agent types
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data from data agents"""
        response = await self.send_message(
            recipient_id="market_data_agent",
            message_type="get_market_data",
            payload={'symbol': symbol},
            wait_for_reply=True
        )
        return response
    
    async def get_ml_prediction(self, 
                               model_type: str,
                               features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get prediction from ML agents"""
        response = await self.send_message(
            recipient_id=f"{model_type}_agent",
            message_type="predict",
            payload={'features': features},
            wait_for_reply=True
        )
        return response
    
    async def execute_trade(self, trade_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute trade through execution agents"""
        response = await self.send_message(
            recipient_id="execution_agent",
            message_type="execute_trade",
            payload=trade_params,
            wait_for_reply=True,
            priority=MessagePriority.HIGH
        )
        return response 