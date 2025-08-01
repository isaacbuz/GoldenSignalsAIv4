"""
Agent Communication MCP Server
Enables efficient inter-agent communication and coordination
Issue #192: MCP-3: Build Agent Communication MCP Server
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class MessageType(Enum):
    """Types of messages"""
    BROADCAST = "broadcast"
    DIRECT = "direct"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_VOTE = "consensus_vote"


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    id: str
    name: str
    type: str
    capabilities: List[str]
    status: str
    last_seen: datetime
    performance_score: float = 1.0
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'last_seen': self.last_seen.isoformat()
        }


@dataclass
class Message:
    """Message structure for agent communication"""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str]  # None for broadcast
    topic: str
    content: Any
    priority: MessagePriority
    timestamp: datetime
    ttl: Optional[int] = None  # Time to live in seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'sender': self.sender,
            'recipient': self.recipient,
            'topic': self.topic,
            'content': self.content,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'ttl': self.ttl
        }

    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl


@dataclass
class ConsensusSession:
    """Represents a consensus voting session"""
    id: str
    question: str
    participants: List[str]
    votes: Dict[str, Any]
    created_at: datetime
    timeout_ms: int
    status: str
    result: Optional[Any] = None

    def is_expired(self) -> bool:
        """Check if session has timed out"""
        age_ms = (datetime.now() - self.created_at).total_seconds() * 1000
        return age_ms > self.timeout_ms


class AgentCommunicationMCP:
    """
    MCP Server for inter-agent communication and coordination
    Provides pub/sub, direct messaging, and consensus mechanisms
    """

    def __init__(self):
        self.app = FastAPI(title="Agent Communication MCP Server")

        # Agent registry
        self.agents: Dict[str, AgentInfo] = {}

        # Topic subscriptions
        self.topic_subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # Message queues
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Active WebSocket connections
        self.websocket_connections: Dict[str, WebSocket] = {}

        # Consensus sessions
        self.consensus_sessions: Dict[str, ConsensusSession] = {}

        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'consensus_sessions': 0,
            'active_agents': 0
        }

        self._setup_routes()

        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._message_cleanup())

    def _setup_routes(self):
        """Set up FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "service": "Agent Communication MCP",
                "status": "active",
                "metrics": self.metrics
            }

        @self.app.get("/tools")
        async def list_tools():
            """List available communication tools"""
            return {
                "tools": [
                    {
                        "name": "register_agent",
                        "description": "Register an agent with the communication system",
                        "parameters": {
                            "id": "string (unique agent ID)",
                            "name": "string",
                            "type": "string",
                            "capabilities": "array[string]"
                        }
                    },
                    {
                        "name": "broadcast",
                        "description": "Broadcast a message to all agents subscribed to a topic",
                        "parameters": {
                            "topic": "string",
                            "sender": "string",
                            "content": "any",
                            "priority": "string (critical/high/normal/low)"
                        }
                    },
                    {
                        "name": "send_direct",
                        "description": "Send a direct message to a specific agent",
                        "parameters": {
                            "sender": "string",
                            "recipient": "string",
                            "content": "any",
                            "priority": "string"
                        }
                    },
                    {
                        "name": "subscribe",
                        "description": "Subscribe an agent to a topic",
                        "parameters": {
                            "agent_id": "string",
                            "topic": "string"
                        }
                    },
                    {
                        "name": "create_consensus",
                        "description": "Create a consensus voting session",
                        "parameters": {
                            "question": "string",
                            "participants": "array[string]",
                            "timeout_ms": "integer"
                        }
                    },
                    {
                        "name": "vote",
                        "description": "Submit a vote in a consensus session",
                        "parameters": {
                            "session_id": "string",
                            "agent_id": "string",
                            "vote": "any",
                            "confidence": "number (0-1)"
                        }
                    }
                ]
            }

        @self.app.post("/call")
        async def call_tool(request: Dict[str, Any], background_tasks: BackgroundTasks):
            """Execute a communication tool"""
            tool_name = request.get("tool")
            params = request.get("parameters", {})

            try:
                if tool_name == "register_agent":
                    return await self._register_agent(params)
                elif tool_name == "broadcast":
                    return await self._broadcast_message(params, background_tasks)
                elif tool_name == "send_direct":
                    return await self._send_direct_message(params, background_tasks)
                elif tool_name == "subscribe":
                    return await self._subscribe_to_topic(params)
                elif tool_name == "create_consensus":
                    return await self._create_consensus_session(params)
                elif tool_name == "vote":
                    return await self._submit_vote(params)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return {"error": str(e), "tool": tool_name}

        @self.app.get("/agents")
        async def list_agents():
            """List all registered agents"""
            return {
                "agents": [agent.to_dict() for agent in self.agents.values()],
                "count": len(self.agents)
            }

        @self.app.get("/topics")
        async def list_topics():
            """List all topics and their subscribers"""
            return {
                "topics": {
                    topic: list(subscribers)
                    for topic, subscribers in self.topic_subscriptions.items()
                }
            }

        @self.app.websocket("/ws/{agent_id}")
        async def websocket_endpoint(websocket: WebSocket, agent_id: str):
            """WebSocket endpoint for real-time agent communication"""
            await websocket.accept()
            self.websocket_connections[agent_id] = websocket

            try:
                # Send any queued messages
                await self._flush_message_queue(agent_id)

                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    request = json.loads(data)

                    if request.get('type') == 'heartbeat':
                        await self._update_agent_heartbeat(agent_id)
                        await websocket.send_json({"type": "heartbeat_ack"})
                    elif request.get('type') == 'message':
                        await self._handle_websocket_message(agent_id, request)

            except Exception as e:
                logger.error(f"WebSocket error for agent {agent_id}: {e}")
            finally:
                del self.websocket_connections[agent_id]
                await websocket.close()

    async def _register_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new agent"""
        agent_id = params.get('id')
        if not agent_id:
            raise ValueError("Agent ID is required")

        agent = AgentInfo(
            id=agent_id,
            name=params.get('name', agent_id),
            type=params.get('type', 'unknown'),
            capabilities=params.get('capabilities', []),
            status='active',
            last_seen=datetime.now()
        )

        self.agents[agent_id] = agent
        self.metrics['active_agents'] = len(self.agents)

        logger.info(f"Registered agent: {agent_id}")

        return {
            "status": "registered",
            "agent": agent.to_dict()
        }

    async def _broadcast_message(self, params: Dict[str, Any],
                               background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Broadcast a message to topic subscribers"""
        topic = params.get('topic')
        if not topic:
            raise ValueError("Topic is required")

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.BROADCAST,
            sender=params.get('sender', 'system'),
            recipient=None,
            topic=topic,
            content=params.get('content', {}),
            priority=MessagePriority[params.get('priority', 'NORMAL').upper()],
            timestamp=datetime.now(),
            ttl=params.get('ttl')
        )

        # Get subscribers
        subscribers = self.topic_subscriptions.get(topic, set())

        # Send message asynchronously
        background_tasks.add_task(self._deliver_message, message, list(subscribers))

        self.metrics['messages_sent'] += 1

        return {
            "status": "broadcast",
            "message_id": message.id,
            "topic": topic,
            "subscriber_count": len(subscribers)
        }

    async def _send_direct_message(self, params: Dict[str, Any],
                                 background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Send a direct message to a specific agent"""
        recipient = params.get('recipient')
        if not recipient:
            raise ValueError("Recipient is required")

        if recipient not in self.agents:
            raise ValueError(f"Unknown agent: {recipient}")

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.DIRECT,
            sender=params.get('sender', 'system'),
            recipient=recipient,
            topic='direct',
            content=params.get('content', {}),
            priority=MessagePriority[params.get('priority', 'NORMAL').upper()],
            timestamp=datetime.now(),
            ttl=params.get('ttl')
        )

        # Send message asynchronously
        background_tasks.add_task(self._deliver_message, message, [recipient])

        self.metrics['messages_sent'] += 1

        return {
            "status": "sent",
            "message_id": message.id,
            "recipient": recipient
        }

    async def _subscribe_to_topic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Subscribe an agent to a topic"""
        agent_id = params.get('agent_id')
        topic = params.get('topic')

        if not all([agent_id, topic]):
            raise ValueError("Agent ID and topic are required")

        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        self.topic_subscriptions[topic].add(agent_id)

        return {
            "status": "subscribed",
            "agent_id": agent_id,
            "topic": topic,
            "total_subscribers": len(self.topic_subscriptions[topic])
        }

    async def _create_consensus_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new consensus voting session"""
        session_id = str(uuid.uuid4())

        session = ConsensusSession(
            id=session_id,
            question=params.get('question', ''),
            participants=params.get('participants', []),
            votes={},
            created_at=datetime.now(),
            timeout_ms=params.get('timeout_ms', 5000),
            status='voting'
        )

        self.consensus_sessions[session_id] = session
        self.metrics['consensus_sessions'] += 1

        # Notify participants
        notification = Message(
            id=str(uuid.uuid4()),
            type=MessageType.CONSENSUS_REQUEST,
            sender='consensus_system',
            recipient=None,
            topic='consensus',
            content={
                'session_id': session_id,
                'question': session.question,
                'timeout_ms': session.timeout_ms
            },
            priority=MessagePriority.HIGH,
            timestamp=datetime.now()
        )

        await self._deliver_message(notification, session.participants)

        # Schedule result calculation
        asyncio.create_task(self._process_consensus_session(session_id))

        return {
            "status": "created",
            "session_id": session_id,
            "participants": len(session.participants)
        }

    async def _submit_vote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a vote in a consensus session"""
        session_id = params.get('session_id')
        agent_id = params.get('agent_id')

        if session_id not in self.consensus_sessions:
            raise ValueError(f"Unknown session: {session_id}")

        session = self.consensus_sessions[session_id]

        if session.status != 'voting':
            raise ValueError(f"Session is not accepting votes: {session.status}")

        if agent_id not in session.participants:
            raise ValueError(f"Agent {agent_id} is not a participant")

        session.votes[agent_id] = {
            'value': params.get('vote'),
            'confidence': params.get('confidence', 1.0),
            'timestamp': datetime.now().isoformat()
        }

        return {
            "status": "vote_recorded",
            "session_id": session_id,
            "votes_received": len(session.votes),
            "votes_needed": len(session.participants)
        }

    async def _deliver_message(self, message: Message, recipients: List[str]):
        """Deliver a message to recipients"""
        delivered_count = 0

        for recipient in recipients:
            # Try WebSocket first
            if recipient in self.websocket_connections:
                try:
                    ws = self.websocket_connections[recipient]
                    await ws.send_json(message.to_dict())
                    delivered_count += 1
                except Exception as e:
                    logger.error(f"Failed to deliver via WebSocket to {recipient}: {e}")
                    # Fall back to queue
                    self.message_queues[recipient].append(message)
            else:
                # Queue for later delivery
                self.message_queues[recipient].append(message)

        self.metrics['messages_delivered'] += delivered_count

        # Update agent activity
        if message.sender in self.agents:
            self.agents[message.sender].message_count += 1

    async def _flush_message_queue(self, agent_id: str):
        """Send all queued messages to an agent"""
        if agent_id not in self.websocket_connections:
            return

        ws = self.websocket_connections[agent_id]
        queue = self.message_queues[agent_id]

        while queue:
            message = queue.popleft()
            if not message.is_expired():
                try:
                    await ws.send_json(message.to_dict())
                    self.metrics['messages_delivered'] += 1
                except Exception as e:
                    logger.error(f"Failed to deliver queued message: {e}")
                    queue.appendleft(message)  # Put it back
                    break

    async def _process_consensus_session(self, session_id: str):
        """Process a consensus session after timeout"""
        session = self.consensus_sessions.get(session_id)
        if not session:
            return

        # Wait for timeout
        await asyncio.sleep(session.timeout_ms / 1000)

        # Calculate result
        session.status = 'completed'

        # Get agent weights based on performance
        weighted_votes = {}
        total_weight = 0

        for agent_id, vote in session.votes.items():
            agent = self.agents.get(agent_id)
            weight = agent.performance_score if agent else 1.0
            weight *= vote['confidence']

            vote_value = str(vote['value'])
            if vote_value not in weighted_votes:
                weighted_votes[vote_value] = 0
            weighted_votes[vote_value] += weight
            total_weight += weight

        # Determine winner
        if weighted_votes:
            winner = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[winner] / total_weight if total_weight > 0 else 0

            session.result = {
                'consensus': winner,
                'confidence': confidence,
                'votes': session.votes,
                'participation_rate': len(session.votes) / len(session.participants)
            }
        else:
            session.result = {
                'consensus': None,
                'confidence': 0,
                'votes': {},
                'participation_rate': 0
            }

        # Notify participants of result
        result_message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            sender='consensus_system',
            recipient=None,
            topic='consensus_result',
            content={
                'session_id': session_id,
                'result': session.result
            },
            priority=MessagePriority.HIGH,
            timestamp=datetime.now()
        )

        await self._deliver_message(result_message, session.participants)

    async def _update_agent_heartbeat(self, agent_id: str):
        """Update agent's last seen timestamp"""
        if agent_id in self.agents:
            self.agents[agent_id].last_seen = datetime.now()
            self.agents[agent_id].status = 'active'

    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and mark inactive agents"""
        while True:
            try:
                now = datetime.now()
                inactive_threshold = timedelta(minutes=5)

                for agent in self.agents.values():
                    if now - agent.last_seen > inactive_threshold:
                        agent.status = 'inactive'

                self.metrics['active_agents'] = sum(
                    1 for a in self.agents.values() if a.status == 'active'
                )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(60)

    async def _message_cleanup(self):
        """Clean up expired messages from queues"""
        while True:
            try:
                for agent_id, queue in self.message_queues.items():
                    # Remove expired messages
                    new_queue = deque(
                        msg for msg in queue if not msg.is_expired()
                    )
                    self.message_queues[agent_id] = new_queue

                # Clean up old consensus sessions
                expired_sessions = [
                    sid for sid, session in self.consensus_sessions.items()
                    if session.is_expired() and session.status == 'completed'
                ]
                for sid in expired_sessions:
                    del self.consensus_sessions[sid]

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                logger.error(f"Message cleanup error: {e}")
                await asyncio.sleep(300)

    def get_agent_weight(self, agent_id: str) -> float:
        """Get agent's weight for consensus based on performance"""
        agent = self.agents.get(agent_id)
        if not agent:
            return 1.0

        # Simple weighting based on performance score and activity
        weight = agent.performance_score

        # Boost for active agents
        if agent.status == 'active':
            weight *= 1.2

        # Penalty for low activity
        if agent.message_count < 10:
            weight *= 0.8

        return max(0.1, min(2.0, weight))


# Demo function
async def demo_agent_communication():
    """Demonstrate the Agent Communication MCP Server"""

    # Create the server
    server = AgentCommunicationMCP()

    print("Agent Communication MCP Demo")
    print("="*60)

    # Register some agents
    agents = [
        {"id": "regime_agent", "name": "Market Regime Agent", "type": "analysis",
         "capabilities": ["regime_classification", "volatility_prediction"]},
        {"id": "liquidity_agent", "name": "Liquidity Agent", "type": "execution",
         "capabilities": ["liquidity_prediction", "execution_timing"]},
        {"id": "news_agent", "name": "News Agent", "type": "sentiment",
         "capabilities": ["news_analysis", "sentiment_scoring"]},
        {"id": "risk_agent", "name": "Risk Agent", "type": "risk",
         "capabilities": ["risk_assessment", "position_sizing"]}
    ]

    print("\n1. Registering Agents:")
    for agent_params in agents:
        result = await server._register_agent(agent_params)
        print(f"   ✓ {agent_params['name']}")

    # Subscribe agents to topics
    print("\n2. Setting up Topic Subscriptions:")
    subscriptions = [
        ("regime_agent", "market_update"),
        ("liquidity_agent", "market_update"),
        ("news_agent", "news_feed"),
        ("risk_agent", "risk_alert"),
        ("risk_agent", "market_update")
    ]

    for agent_id, topic in subscriptions:
        await server._subscribe_to_topic({"agent_id": agent_id, "topic": topic})
        print(f"   ✓ {agent_id} → {topic}")

    # Test broadcast message
    print("\n3. Broadcasting Market Update:")
    broadcast_result = await server._broadcast_message({
        "topic": "market_update",
        "sender": "market_data_mcp",
        "content": {
            "vix": 25.5,
            "spy_change": -1.2,
            "volume_spike": 1.8
        },
        "priority": "high"
    }, BackgroundTasks())
    print(f"   Sent to {broadcast_result['subscriber_count']} agents")

    # Test direct message
    print("\n4. Sending Direct Message:")
    direct_result = await server._send_direct_message({
        "sender": "regime_agent",
        "recipient": "risk_agent",
        "content": {
            "alert": "Regime change detected",
            "new_regime": "bear",
            "confidence": 0.85
        },
        "priority": "critical"
    }, BackgroundTasks())
    print(f"   Message {direct_result['message_id'][:8]}... sent")

    # Test consensus mechanism
    print("\n5. Creating Consensus Session:")
    consensus_result = await server._create_consensus_session({
        "question": "Should we reduce position size given market conditions?",
        "participants": ["regime_agent", "liquidity_agent", "risk_agent"],
        "timeout_ms": 2000
    })
    session_id = consensus_result['session_id']
    print(f"   Session {session_id[:8]}... created")

    # Submit votes
    print("\n6. Agents Voting:")
    votes = [
        ("regime_agent", "yes", 0.9),
        ("liquidity_agent", "no", 0.6),
        ("risk_agent", "yes", 0.95)
    ]

    for agent_id, vote, confidence in votes:
        await server._submit_vote({
            "session_id": session_id,
            "agent_id": agent_id,
            "vote": vote,
            "confidence": confidence
        })
        print(f"   {agent_id}: {vote} (confidence: {confidence})")

    # Wait for consensus
    await asyncio.sleep(2.5)

    # Check result
    session = server.consensus_sessions[session_id]
    if session.result:
        print(f"\n7. Consensus Result:")
        print(f"   Decision: {session.result['consensus']}")
        print(f"   Confidence: {session.result['confidence']:.1%}")
        print(f"   Participation: {session.result['participation_rate']:.1%}")

    # Show metrics
    print(f"\n8. Communication Metrics:")
    print(f"   Active Agents: {server.metrics['active_agents']}")
    print(f"   Messages Sent: {server.metrics['messages_sent']}")
    print(f"   Consensus Sessions: {server.metrics['consensus_sessions']}")


if __name__ == "__main__":
    asyncio.run(demo_agent_communication())

    # To run as a server:
    # import uvicorn
    # server = AgentCommunicationMCP()
    # uvicorn.run(server.app, host="0.0.0.0", port=8001)
