"""
Horizontal Scaling Architecture for Agents
Enables distributed agent execution across multiple instances
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status in cluster"""

    ACTIVE = "active"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class ClusterNode:
    """Represents a node in the cluster"""

    node_id: str
    hostname: str
    port: int
    status: NodeStatus
    capacity: int  # Max concurrent agents
    current_load: int
    last_heartbeat: datetime
    capabilities: Set[str]  # Agent types this node can handle


class HorizontalScalingManager:
    """Manages horizontal scaling for agent execution"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.node_id = self._generate_node_id()
        self.nodes: Dict[str, ClusterNode] = {}
        self.agent_assignments: Dict[str, str] = {}  # agent_id -> node_id

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import socket

        hostname = socket.gethostname()
        timestamp = datetime.now().timestamp()
        return hashlib.md5(f"{hostname}_{timestamp}".encode()).hexdigest()[:12]

    async def initialize(self):
        """Initialize scaling manager"""
        self.redis_client = await redis.from_url(self.redis_url)

        # Register this node
        await self._register_node()

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._monitor_cluster())

        logger.info(f"Horizontal scaling manager initialized. Node ID: {self.node_id}")

    async def _register_node(self):
        """Register this node in the cluster"""
        node = ClusterNode(
            node_id=self.node_id,
            hostname="localhost",  # In production, use actual hostname
            port=8000,  # In production, use actual port
            status=NodeStatus.ACTIVE,
            capacity=10,  # Max agents per node
            current_load=0,
            last_heartbeat=datetime.now(),
            capabilities={"sentiment", "technical", "flow", "risk", "regime"},
        )

        node_data = {
            "node_id": node.node_id,
            "hostname": node.hostname,
            "port": node.port,
            "status": node.status.value,
            "capacity": node.capacity,
            "current_load": node.current_load,
            "last_heartbeat": node.last_heartbeat.isoformat(),
            "capabilities": list(node.capabilities),
        }

        await self.redis_client.hset("cluster:nodes", node.node_id, json.dumps(node_data))

        self.nodes[node.node_id] = node

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await self._update_heartbeat()
                await asyncio.sleep(5)  # Heartbeat every 5 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)

    async def _update_heartbeat(self):
        """Update node heartbeat"""
        if self.node_id in self.nodes:
            node = self.nodes[self.node_id]
            node.last_heartbeat = datetime.now()

            # Update Redis
            node_data = await self.redis_client.hget("cluster:nodes", self.node_id)
            if node_data:
                data = json.loads(node_data)
                data["last_heartbeat"] = node.last_heartbeat.isoformat()
                data["current_load"] = node.current_load

                await self.redis_client.hset("cluster:nodes", self.node_id, json.dumps(data))

    async def _monitor_cluster(self):
        """Monitor cluster health"""
        while True:
            try:
                await self._check_node_health()
                await self._rebalance_if_needed()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Cluster monitoring error: {e}")
                await asyncio.sleep(5)

    async def _check_node_health(self):
        """Check health of all nodes"""
        nodes_data = await self.redis_client.hgetall("cluster:nodes")

        for node_id, node_data in nodes_data.items():
            node_id = node_id.decode() if isinstance(node_id, bytes) else node_id
            data = json.loads(node_data)

            last_heartbeat = datetime.fromisoformat(data["last_heartbeat"])

            # Mark node as offline if no heartbeat for 30 seconds
            if datetime.now() - last_heartbeat > timedelta(seconds=30):
                data["status"] = NodeStatus.OFFLINE.value
                await self.redis_client.hset("cluster:nodes", node_id, json.dumps(data))

                # Reassign agents from offline node
                if node_id in self.nodes and self.nodes[node_id].status != NodeStatus.OFFLINE:
                    await self._handle_node_failure(node_id)

            # Update local node cache
            self.nodes[node_id] = ClusterNode(
                node_id=node_id,
                hostname=data["hostname"],
                port=data["port"],
                status=NodeStatus(data["status"]),
                capacity=data["capacity"],
                current_load=data["current_load"],
                last_heartbeat=last_heartbeat,
                capabilities=set(data["capabilities"]),
            )

    async def _handle_node_failure(self, failed_node_id: str):
        """Handle node failure by reassigning agents"""
        logger.warning(f"Node {failed_node_id} failed. Reassigning agents...")

        # Find agents assigned to failed node
        failed_agents = [
            agent_id
            for agent_id, node_id in self.agent_assignments.items()
            if node_id == failed_node_id
        ]

        # Reassign agents
        for agent_id in failed_agents:
            new_node = await self._find_best_node(agent_type="general")
            if new_node:
                self.agent_assignments[agent_id] = new_node.node_id
                await self._notify_agent_reassignment(agent_id, new_node.node_id)

    async def _rebalance_if_needed(self):
        """Rebalance load across nodes if needed"""
        active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]

        if len(active_nodes) < 2:
            return

        # Calculate load variance
        loads = [n.current_load / n.capacity for n in active_nodes]
        avg_load = sum(loads) / len(loads)
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)

        # Rebalance if variance is high
        if variance > 0.1:  # 10% variance threshold
            await self._rebalance_cluster()

    async def _rebalance_cluster(self):
        """Rebalance agents across cluster"""
        logger.info("Rebalancing cluster load...")

        # This is a simplified rebalancing algorithm
        # In production, use more sophisticated algorithms

        active_nodes = sorted(
            [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE],
            key=lambda n: n.current_load / n.capacity,
        )

        if len(active_nodes) < 2:
            return

        # Move agents from most loaded to least loaded
        most_loaded = active_nodes[-1]
        least_loaded = active_nodes[0]

        if most_loaded.current_load > least_loaded.current_load + 2:
            # Find an agent to move
            for agent_id, node_id in self.agent_assignments.items():
                if node_id == most_loaded.node_id:
                    # Move agent
                    self.agent_assignments[agent_id] = least_loaded.node_id
                    most_loaded.current_load -= 1
                    least_loaded.current_load += 1

                    await self._notify_agent_reassignment(agent_id, least_loaded.node_id)
                    break

    async def assign_agent(self, agent_id: str, agent_type: str) -> Optional[str]:
        """Assign agent to best available node"""
        node = await self._find_best_node(agent_type)

        if node:
            self.agent_assignments[agent_id] = node.node_id
            node.current_load += 1

            # Update Redis
            await self.redis_client.hset("cluster:assignments", agent_id, node.node_id)

            logger.info(f"Assigned agent {agent_id} to node {node.node_id}")
            return node.node_id

        logger.error(f"No available node for agent {agent_id}")
        return None

    async def _find_best_node(self, agent_type: str) -> Optional[ClusterNode]:
        """Find best node for agent type"""
        suitable_nodes = [
            n
            for n in self.nodes.values()
            if n.status == NodeStatus.ACTIVE
            and agent_type in n.capabilities
            and n.current_load < n.capacity
        ]

        if not suitable_nodes:
            return None

        # Choose node with lowest load ratio
        return min(suitable_nodes, key=lambda n: n.current_load / n.capacity)

    async def release_agent(self, agent_id: str):
        """Release agent assignment"""
        if agent_id in self.agent_assignments:
            node_id = self.agent_assignments[agent_id]

            if node_id in self.nodes:
                self.nodes[node_id].current_load -= 1

            del self.agent_assignments[agent_id]

            await self.redis_client.hdel("cluster:assignments", agent_id)

            logger.info(f"Released agent {agent_id}")

    async def _notify_agent_reassignment(self, agent_id: str, new_node_id: str):
        """Notify about agent reassignment"""
        await self.redis_client.publish(
            "cluster:reassignments",
            json.dumps(
                {
                    "agent_id": agent_id,
                    "new_node_id": new_node_id,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
        )

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
        total_capacity = sum(n.capacity for n in active_nodes)
        total_load = sum(n.current_load for n in active_nodes)

        return {
            "nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "total_capacity": total_capacity,
            "total_load": total_load,
            "load_percentage": (total_load / total_capacity * 100) if total_capacity > 0 else 0,
            "node_details": [
                {
                    "node_id": n.node_id,
                    "status": n.status.value,
                    "load": f"{n.current_load}/{n.capacity}",
                    "capabilities": list(n.capabilities),
                }
                for n in self.nodes.values()
            ],
        }

    async def drain_node(self, node_id: Optional[str] = None):
        """Drain a node for maintenance"""
        target_node_id = node_id or self.node_id

        if target_node_id in self.nodes:
            node = self.nodes[target_node_id]
            node.status = NodeStatus.DRAINING

            # Update Redis
            node_data = await self.redis_client.hget("cluster:nodes", target_node_id)
            if node_data:
                data = json.loads(node_data)
                data["status"] = NodeStatus.DRAINING.value

                await self.redis_client.hset("cluster:nodes", target_node_id, json.dumps(data))

            # Reassign agents from draining node
            await self._handle_node_failure(target_node_id)

            logger.info(f"Node {target_node_id} set to draining")


# Global scaling manager
scaling_manager = HorizontalScalingManager()
