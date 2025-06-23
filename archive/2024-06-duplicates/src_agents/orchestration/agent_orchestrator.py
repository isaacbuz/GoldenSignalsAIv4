"""
Agent Orchestrator for GoldenSignalsAI
Central coordination system for all agents
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from enum import Enum

import ray
from ray import serve
import networkx as nx
from prometheus_client import Counter, Histogram, Gauge

from agents.core.unified_base_agent import UnifiedBaseAgent, AgentType, MessagePriority, AgentMessage


# Metrics
orchestration_requests = Counter('orchestration_requests_total', 'Total orchestration requests', ['strategy'])
orchestration_latency = Histogram('orchestration_latency_seconds', 'Orchestration latency', ['strategy'])
active_agents = Gauge('active_agents_total', 'Number of active agents', ['agent_type'])
agent_utilization = Gauge('agent_utilization_ratio', 'Agent utilization', ['agent_id'])


class OrchestrationStrategy(Enum):
    """Different orchestration strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ENSEMBLE = "ensemble"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class AgentTask:
    """Task to be executed by an agent"""
    task_id: str
    agent_id: str
    task_type: str
    params: Dict[str, Any]
    dependencies: List[str] = None  # Task IDs this depends on
    priority: MessagePriority = MessagePriority.NORMAL
    timeout: float = 30.0
    retry_count: int = 3


@dataclass
class WorkflowDefinition:
    """Defines a workflow of agent tasks"""
    workflow_id: str
    name: str
    tasks: List[AgentTask]
    strategy: OrchestrationStrategy
    timeout: float = 300.0
    metadata: Dict[str, Any] = None


class AgentOrchestrator(UnifiedBaseAgent):
    """Master orchestrator for coordinating all agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.ORCHESTRATOR, config)
        
        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_results: Dict[str, Dict[str, Any]] = {}
        
        # Task execution
        self.task_queue = asyncio.Queue()
        self.executing_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Agent selection
        self.agent_graph = nx.DiGraph()  # For hierarchical strategies
        self.load_balancer = self._create_load_balancer()
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
    def _register_capabilities(self):
        """Register orchestrator capabilities"""
        self.capabilities = {
            'orchestrate': {
                'description': 'Orchestrate multi-agent workflows',
                'input': {'workflow': 'WorkflowDefinition'},
                'output': {'results': 'Dict[str, Any]'}
            },
            'optimize_workflow': {
                'description': 'Optimize workflow execution',
                'input': {'workflow_id': 'str'},
                'output': {'optimized_workflow': 'WorkflowDefinition'}
            },
            'get_agent_status': {
                'description': 'Get status of all agents',
                'input': {},
                'output': {'agents': 'List[Dict]'}
            }
        }
    
    def _register_message_handlers(self):
        """Register message handlers"""
        self.message_handlers = {
            'register_agent': self.handle_agent_registration,
            'agent_heartbeat': self.handle_agent_heartbeat,
            'execute_workflow': self.handle_execute_workflow,
            'task_completed': self.handle_task_completed,
            'get_recommendation': self.handle_get_recommendation
        }
    
    async def handle_agent_registration(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle agent registration"""
        agent_info = message.payload
        agent_id = agent_info['agent_id']
        
        self.registered_agents[agent_id] = {
            'info': agent_info,
            'last_seen': datetime.now(),
            'status': 'active',
            'performance': {'success_rate': 1.0, 'avg_latency': 0.0}
        }
        
        # Update capabilities index
        for capability in agent_info.get('capabilities', []):
            if capability not in self.agent_capabilities:
                self.agent_capabilities[capability] = []
            self.agent_capabilities[capability].append(agent_id)
        
        # Update metrics
        agent_type = agent_info.get('agent_type', 'unknown')
        active_agents.labels(agent_type=agent_type).inc()
        
        self.logger.info(f"Registered agent: {agent_id}")
        
        return {'status': 'registered', 'agent_id': agent_id}
    
    async def handle_execute_workflow(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a workflow"""
        workflow_def = WorkflowDefinition(**message.payload['workflow'])
        
        # Track metrics
        orchestration_requests.labels(strategy=workflow_def.strategy.value).inc()
        
        # Store workflow
        self.active_workflows[workflow_def.workflow_id] = workflow_def
        
        # Execute based on strategy
        with orchestration_latency.labels(strategy=workflow_def.strategy.value).time():
            if workflow_def.strategy == OrchestrationStrategy.SEQUENTIAL:
                result = await self._execute_sequential(workflow_def)
            elif workflow_def.strategy == OrchestrationStrategy.PARALLEL:
                result = await self._execute_parallel(workflow_def)
            elif workflow_def.strategy == OrchestrationStrategy.ENSEMBLE:
                result = await self._execute_ensemble(workflow_def)
            elif workflow_def.strategy == OrchestrationStrategy.HIERARCHICAL:
                result = await self._execute_hierarchical(workflow_def)
            elif workflow_def.strategy == OrchestrationStrategy.ADAPTIVE:
                result = await self._execute_adaptive(workflow_def)
            else:
                result = {'error': f'Unknown strategy: {workflow_def.strategy}'}
        
        # Store results
        self.workflow_results[workflow_def.workflow_id] = result
        
        # Clean up
        del self.active_workflows[workflow_def.workflow_id]
        
        return result
    
    async def _execute_sequential(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        results = {}
        
        for task in workflow.tasks:
            # Wait for dependencies
            if task.dependencies:
                for dep_id in task.dependencies:
                    while dep_id not in results:
                        await asyncio.sleep(0.1)
            
            # Execute task
            result = await self._execute_task(task)
            results[task.task_id] = result
            
            # Check for errors
            if result.get('error'):
                self.logger.error(f"Task {task.task_id} failed: {result['error']}")
                if not workflow.metadata.get('continue_on_error', False):
                    break
        
        return {
            'workflow_id': workflow.workflow_id,
            'strategy': 'sequential',
            'results': results,
            'completed': len(results) == len(workflow.tasks)
        }
    
    async def _execute_parallel(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute tasks in parallel"""
        # Group tasks by dependencies
        task_groups = self._group_tasks_by_dependencies(workflow.tasks)
        results = {}
        
        # Execute each group in order
        for group in task_groups:
            # Execute all tasks in group parallel
            group_tasks = []
            for task in group:
                group_tasks.append(self._execute_task(task))
            
            # Wait for all tasks in group
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # Store results
            for task, result in zip(group, group_results):
                if isinstance(result, Exception):
                    results[task.task_id] = {'error': str(result)}
                else:
                    results[task.task_id] = result
        
        return {
            'workflow_id': workflow.workflow_id,
            'strategy': 'parallel',
            'results': results,
            'completed': len(results) == len(workflow.tasks)
        }
    
    async def _execute_ensemble(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute ensemble strategy - multiple agents vote on outcome"""
        # Group tasks by type
        task_groups = {}
        for task in workflow.tasks:
            task_type = task.task_type
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(task)
        
        ensemble_results = {}
        
        for task_type, tasks in task_groups.items():
            # Execute all tasks of this type in parallel
            task_futures = [self._execute_task(task) for task in tasks]
            results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            # Filter out errors
            valid_results = [r for r in results if not isinstance(r, Exception) and 'error' not in r]
            
            if valid_results:
                # Aggregate results (voting, averaging, etc.)
                ensemble_results[task_type] = self._aggregate_ensemble_results(valid_results)
            else:
                ensemble_results[task_type] = {'error': 'No valid results from ensemble'}
        
        return {
            'workflow_id': workflow.workflow_id,
            'strategy': 'ensemble',
            'results': ensemble_results,
            'ensemble_size': len(workflow.tasks)
        }
    
    def _aggregate_ensemble_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from ensemble execution"""
        # Extract predictions/signals
        predictions = []
        confidences = []
        
        for result in results:
            if 'prediction' in result:
                predictions.append(result['prediction'])
                confidences.append(result.get('confidence', 1.0))
        
        if not predictions:
            return {'error': 'No predictions in results'}
        
        # Weighted average based on confidence
        weights = np.array(confidences)
        weights = weights / weights.sum()
        
        if isinstance(predictions[0], (int, float)):
            # Numeric predictions - weighted average
            ensemble_prediction = np.average(predictions, weights=weights)
        else:
            # Categorical - weighted voting
            from collections import Counter
            vote_counts = Counter()
            for pred, weight in zip(predictions, weights):
                vote_counts[pred] += weight
            ensemble_prediction = vote_counts.most_common(1)[0][0]
        
        return {
            'prediction': ensemble_prediction,
            'confidence': np.mean(confidences),
            'agreement': np.std(predictions) if isinstance(predictions[0], (int, float)) else len(set(predictions)) / len(predictions),
            'ensemble_size': len(predictions)
        }
    
    async def _execute_hierarchical(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute hierarchical strategy - agents organized in hierarchy"""
        # Build execution graph
        exec_graph = nx.DiGraph()
        
        for task in workflow.tasks:
            exec_graph.add_node(task.task_id, task=task)
            if task.dependencies:
                for dep in task.dependencies:
                    exec_graph.add_edge(dep, task.task_id)
        
        # Topological sort for execution order
        try:
            execution_order = list(nx.topological_sort(exec_graph))
        except nx.NetworkXError:
            return {'error': 'Circular dependencies detected in workflow'}
        
        results = {}
        
        # Execute in topological order
        for task_id in execution_order:
            task = exec_graph.nodes[task_id]['task']
            
            # Get inputs from dependencies
            inputs = {}
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id in results:
                        inputs[dep_id] = results[dep_id]
            
            # Add dependency results to task params
            task.params['dependency_results'] = inputs
            
            # Execute task
            result = await self._execute_task(task)
            results[task_id] = result
        
        return {
            'workflow_id': workflow.workflow_id,
            'strategy': 'hierarchical',
            'results': results,
            'execution_order': execution_order
        }
    
    async def _execute_adaptive(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute adaptive strategy - dynamically adjust based on results"""
        results = {}
        remaining_tasks = workflow.tasks.copy()
        executed_tasks = set()
        
        while remaining_tasks:
            # Select next best task based on current state
            next_task = await self._select_next_adaptive_task(
                remaining_tasks, 
                results, 
                workflow.metadata
            )
            
            if not next_task:
                break
            
            # Execute selected task
            result = await self._execute_task(next_task)
            results[next_task.task_id] = result
            
            # Update task lists
            remaining_tasks.remove(next_task)
            executed_tasks.add(next_task.task_id)
            
            # Adapt strategy based on result
            if result.get('confidence', 1.0) < 0.5:
                # Low confidence - might need to execute more tasks
                self.logger.info(f"Low confidence result from {next_task.task_id}, adapting strategy")
                
                # Find similar tasks to execute
                similar_tasks = [
                    t for t in remaining_tasks 
                    if t.task_type == next_task.task_type
                ][:2]  # Execute up to 2 more similar tasks
                
                for task in similar_tasks:
                    additional_result = await self._execute_task(task)
                    results[task.task_id] = additional_result
                    remaining_tasks.remove(task)
                    executed_tasks.add(task.task_id)
        
        return {
            'workflow_id': workflow.workflow_id,
            'strategy': 'adaptive',
            'results': results,
            'tasks_executed': len(executed_tasks),
            'tasks_skipped': len(remaining_tasks)
        }
    
    async def _select_next_adaptive_task(self, 
                                       remaining_tasks: List[AgentTask],
                                       current_results: Dict[str, Any],
                                       metadata: Dict[str, Any]) -> Optional[AgentTask]:
        """Select next task for adaptive execution"""
        if not remaining_tasks:
            return None
        
        # Score each task
        task_scores = []
        
        for task in remaining_tasks:
            score = 0
            
            # Check if dependencies are satisfied
            if task.dependencies:
                if not all(dep in current_results for dep in task.dependencies):
                    continue  # Skip tasks with unsatisfied dependencies
            
            # Priority score
            score += (5 - task.priority.value) * 10
            
            # Agent performance score
            agent_perf = self.agent_performance.get(task.agent_id, {})
            score += agent_perf.get('success_rate', 0.5) * 20
            
            # Task type diversity (prefer different types)
            executed_types = [
                t.task_type for t_id, t in self.executing_tasks.items() 
                if t_id in current_results
            ]
            if task.task_type not in executed_types:
                score += 15
            
            task_scores.append((score, task))
        
        if not task_scores:
            return None
        
        # Select task with highest score
        task_scores.sort(key=lambda x: x[0], reverse=True)
        return task_scores[0][1]
    
    async def _execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a single task"""
        self.executing_tasks[task.task_id] = task
        
        try:
            # Send task to agent
            result = await self.send_message(
                recipient_id=task.agent_id,
                message_type=task.task_type,
                payload=task.params,
                priority=task.priority,
                wait_for_reply=True,
                timeout=task.timeout
            )
            
            if result is None:
                result = {'error': 'Task timeout'}
            
            # Update agent performance
            self._update_agent_performance(task.agent_id, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            return {'error': str(e)}
        finally:
            del self.executing_tasks[task.task_id]
    
    def _update_agent_performance(self, agent_id: str, result: Dict[str, Any]):
        """Update agent performance metrics"""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'success_rate': 1.0,
                'avg_latency': 0.0,
                'total_tasks': 0
            }
        
        perf = self.agent_performance[agent_id]
        perf['total_tasks'] += 1
        
        # Update success rate
        if 'error' not in result:
            current_successes = perf['success_rate'] * (perf['total_tasks'] - 1)
            perf['success_rate'] = (current_successes + 1) / perf['total_tasks']
        else:
            current_successes = perf['success_rate'] * (perf['total_tasks'] - 1)
            perf['success_rate'] = current_successes / perf['total_tasks']
    
    def _group_tasks_by_dependencies(self, tasks: List[AgentTask]) -> List[List[AgentTask]]:
        """Group tasks by dependency levels"""
        # Build dependency graph
        dep_graph = nx.DiGraph()
        
        for task in tasks:
            dep_graph.add_node(task.task_id, task=task)
            if task.dependencies:
                for dep in task.dependencies:
                    dep_graph.add_edge(dep, task.task_id)
        
        # Find tasks at each level
        levels = []
        remaining_nodes = set(dep_graph.nodes())
        
        while remaining_nodes:
            # Find nodes with no dependencies in remaining set
            current_level = []
            for node in remaining_nodes:
                predecessors = set(dep_graph.predecessors(node))
                if not predecessors.intersection(remaining_nodes):
                    current_level.append(dep_graph.nodes[node]['task'])
            
            if not current_level:
                # Circular dependency
                break
            
            levels.append(current_level)
            remaining_nodes -= set(t.task_id for t in current_level)
        
        return levels
    
    def _create_load_balancer(self):
        """Create load balancer for agent selection"""
        # Simple round-robin for now
        return {'index': 0}
    
    async def select_best_agent(self, capability: str, context: Dict[str, Any]) -> Optional[str]:
        """Select best agent for a capability"""
        available_agents = self.agent_capabilities.get(capability, [])
        
        if not available_agents:
            return None
        
        # Filter by active agents
        active = [
            agent_id for agent_id in available_agents
            if self.registered_agents.get(agent_id, {}).get('status') == 'active'
        ]
        
        if not active:
            return None
        
        # Select based on performance
        best_agent = None
        best_score = -1
        
        for agent_id in active:
            perf = self.agent_performance.get(agent_id, {})
            score = perf.get('success_rate', 0.5) * (1 - perf.get('avg_latency', 0) / 100)
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    async def handle_get_recommendation(self, message: AgentMessage) -> Dict[str, Any]:
        """Get trading recommendation by orchestrating multiple agents"""
        symbol = message.payload['symbol']
        timeframe = message.payload.get('timeframe', '1h')
        
        # Define workflow for getting comprehensive recommendation
        workflow = WorkflowDefinition(
            workflow_id=f"recommendation_{symbol}_{datetime.now().timestamp()}",
            name=f"Get recommendation for {symbol}",
            strategy=OrchestrationStrategy.ENSEMBLE,
            tasks=[
                # Technical analysis
                AgentTask(
                    task_id="tech_1",
                    agent_id="technical_agent_1",
                    task_type="analyze",
                    params={'symbol': symbol, 'timeframe': timeframe}
                ),
                # Sentiment analysis
                AgentTask(
                    task_id="sentiment_1",
                    agent_id="sentiment_agent_1",
                    task_type="analyze_sentiment",
                    params={'symbol': symbol}
                ),
                # ML predictions
                AgentTask(
                    task_id="ml_lstm",
                    agent_id="lstm_agent_1",
                    task_type="predict",
                    params={'symbol': symbol, 'horizon': 24}
                ),
                AgentTask(
                    task_id="ml_xgboost",
                    agent_id="xgboost_agent_1",
                    task_type="predict",
                    params={'symbol': symbol}
                ),
                # Risk assessment
                AgentTask(
                    task_id="risk_1",
                    agent_id="risk_agent_1",
                    task_type="assess_risk",
                    params={'symbol': symbol}
                )
            ]
        )
        
        # Execute workflow
        result = await self.handle_execute_workflow(AgentMessage(
            id="internal",
            sender_id=self.agent_id,
            recipient_id=self.agent_id,
            message_type="execute_workflow",
            payload={'workflow': workflow.__dict__},
            timestamp=datetime.now()
        ))
        
        # Aggregate results into recommendation
        recommendation = self._create_recommendation(result['results'])
        
        return recommendation
    
    def _create_recommendation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create trading recommendation from agent results"""
        # Aggregate signals
        signals = []
        confidences = []
        
        for task_type, result in results.items():
            if 'signal' in result:
                signals.append(result['signal'])
                confidences.append(result.get('confidence', 0.5))
        
        if not signals:
            return {
                'recommendation': 'HOLD',
                'confidence': 0,
                'reason': 'Insufficient data'
            }
        
        # Weight by confidence
        buy_score = sum(c for s, c in zip(signals, confidences) if s > 0)
        sell_score = sum(c for s, c in zip(signals, confidences) if s < 0)
        
        if buy_score > sell_score * 1.2:  # 20% threshold
            recommendation = 'BUY'
            confidence = buy_score / sum(confidences)
        elif sell_score > buy_score * 1.2:
            recommendation = 'SELL'
            confidence = sell_score / sum(confidences)
        else:
            recommendation = 'HOLD'
            confidence = 1 - abs(buy_score - sell_score) / sum(confidences)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'analysis': results,
            'timestamp': datetime.now().isoformat()
        }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process orchestration request"""
        request_type = request.get('type')
        
        if request_type == 'get_recommendation':
            return await self.handle_get_recommendation(AgentMessage(
                id="api_request",
                sender_id="api",
                recipient_id=self.agent_id,
                message_type="get_recommendation",
                payload=request,
                timestamp=datetime.now()
            ))
        else:
            return {'error': f'Unknown request type: {request_type}'}


# Ray Serve deployment
@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0}
)
class OrchestratorService:
    """Ray Serve deployment for the orchestrator"""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator(
            agent_id="main_orchestrator",
            config={
                'name': 'Main Orchestrator',
                'redis_url': 'redis://localhost',
                'http_port': 8000
            }
        )
        asyncio.create_task(self.orchestrator.run())
    
    async def __call__(self, request):
        """Handle HTTP requests"""
        return await self.orchestrator.process_request(request) 