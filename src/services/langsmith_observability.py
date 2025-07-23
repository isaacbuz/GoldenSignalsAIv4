"""
LangSmith Observability Service
Provides comprehensive monitoring, debugging, and analytics for AI workflows
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from langsmith import Client, RunTree, traceable
from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)


class ObservabilityLevel(Enum):
    """Levels of observability detail"""

    BASIC = "basic"  # Just track inputs/outputs
    DETAILED = "detailed"  # Track intermediate steps
    DEBUG = "debug"  # Full trace with all details
    PRODUCTION = "production"  # Optimized for production


@dataclass
class WorkflowMetrics:
    """Metrics for workflow performance"""

    workflow_name: str
    total_runs: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    cost_per_run: float
    satisfaction_score: Optional[float] = None


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance"""

    agent_name: str
    calls_count: int
    success_rate: float
    avg_confidence: float
    avg_latency_ms: float
    accuracy_score: Optional[float] = None


class LangSmithObservability:
    """
    Comprehensive observability for AI workflows using LangSmith
    """

    def __init__(
        self,
        project_name: str = "goldensignals-trading",
        api_key: Optional[str] = None,
        level: ObservabilityLevel = ObservabilityLevel.DETAILED,
    ):
        self.project_name = project_name
        self.level = level

        # Initialize LangSmith client
        self.client = Client(
            api_key=api_key or os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com"),
        )

        # Create project if it doesn't exist
        self._ensure_project()

        # Metrics cache
        self.metrics_cache = {"workflows": {}, "agents": {}, "last_updated": None}

        # Custom evaluators
        self.evaluators = {}
        self._register_default_evaluators()

    def _ensure_project(self):
        """Ensure the LangSmith project exists"""
        try:
            # Check if project exists
            projects = self.client.list_projects()
            project_exists = any(p.name == self.project_name for p in projects)

            if not project_exists:
                # Create new project
                self.client.create_project(
                    project_name=self.project_name,
                    description="GoldenSignals AI Trading System Observability",
                )
                logger.info(f"Created LangSmith project: {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to ensure project: {e}")

    def trace_workflow(self, name: str, metadata: Optional[Dict] = None):
        """Decorator to trace workflow execution"""

        def decorator(func: Callable):
            @wraps(func)
            @traceable(
                name=name,
                project_name=self.project_name,
                tags=["workflow", self.level.value],
                metadata=metadata or {},
            )
            async def wrapper(*args, **kwargs):
                # Add custom tracking based on level
                if self.level in [ObservabilityLevel.DETAILED, ObservabilityLevel.DEBUG]:
                    # Log inputs
                    logger.info(f"Starting workflow {name} with args: {args}, kwargs: {kwargs}")

                try:
                    result = await func(*args, **kwargs)

                    # Track success
                    if self.level != ObservabilityLevel.BASIC:
                        self._track_workflow_success(name, result)

                    return result

                except Exception as e:
                    # Track error
                    self._track_workflow_error(name, str(e))
                    raise

            return wrapper

        return decorator

    def trace_agent(self, agent_name: str):
        """Decorator to trace agent execution"""

        def decorator(func: Callable):
            @wraps(func)
            @traceable(
                name=f"agent_{agent_name}",
                project_name=self.project_name,
                tags=["agent", agent_name],
                metadata={"agent_type": agent_name},
            )
            async def wrapper(*args, **kwargs):
                start_time = asyncio.get_event_loop().time()

                try:
                    result = await func(*args, **kwargs)

                    # Track agent performance
                    latency = (asyncio.get_event_loop().time() - start_time) * 1000
                    self._track_agent_performance(agent_name, result, latency)

                    return result

                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    raise

            return wrapper

        return decorator

    def trace_llm_call(self, provider: str, model: str):
        """Decorator to trace LLM API calls"""

        def decorator(func: Callable):
            @wraps(func)
            @traceable(
                name=f"llm_{provider}_{model}",
                project_name=self.project_name,
                tags=["llm", provider, model],
            )
            async def wrapper(*args, **kwargs):
                # Extract relevant info for tracking
                prompt = kwargs.get("messages", kwargs.get("prompt", ""))

                try:
                    result = await func(*args, **kwargs)

                    # Track token usage and cost
                    if hasattr(result, "usage"):
                        self._track_llm_usage(provider, model, result.usage)

                    return result

                except Exception as e:
                    logger.error(f"LLM call to {provider}/{model} failed: {e}")
                    raise

            return wrapper

        return decorator

    async def log_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        context: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ):
        """Log a trading decision for analysis"""
        try:
            # Create a run for the decision
            run = RunTree(
                name="trading_decision",
                run_type="chain",
                project_name=self.project_name,
                inputs={"symbol": symbol, "context": context},
                outputs={"decision": decision, "outcome": outcome},
                tags=["decision", symbol],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "confidence": decision.get("confidence", 0),
                    "action": decision.get("action", "HOLD"),
                },
            )

            # Post to LangSmith
            run.post()

            # If outcome is provided, create feedback
            if outcome:
                self._create_feedback(run.id, outcome)

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def _create_feedback(self, run_id: str, outcome: Dict[str, Any]):
        """Create feedback for a run based on outcome"""
        try:
            # Calculate score based on outcome
            score = 1.0 if outcome.get("profitable", False) else 0.0

            # Adjust score based on return
            return_pct = outcome.get("return_pct", 0)
            if return_pct > 0:
                score = min(1.0, 0.5 + (return_pct / 20))  # Max score at 10% return
            else:
                score = max(0.0, 0.5 + (return_pct / 20))  # Min score at -10% loss

            # Create feedback
            self.client.create_feedback(
                run_id=run_id,
                key="trade_outcome",
                score=score,
                comment=f"Return: {return_pct:.2f}%",
            )

        except Exception as e:
            logger.error(f"Failed to create feedback: {e}")

    async def get_workflow_metrics(
        self, workflow_name: Optional[str] = None, time_range: timedelta = timedelta(days=7)
    ) -> List[WorkflowMetrics]:
        """Get metrics for workflows"""
        try:
            # Query runs from LangSmith
            filters = {"start_time": datetime.now() - time_range, "tags": ["workflow"]}

            if workflow_name:
                filters["name"] = workflow_name

            runs = list(
                self.client.list_runs(project_name=self.project_name, filter=filters, limit=1000)
            )

            # Group by workflow name
            workflow_runs = {}
            for run in runs:
                name = run.name
                if name not in workflow_runs:
                    workflow_runs[name] = []
                workflow_runs[name].append(run)

            # Calculate metrics
            metrics = []
            for name, runs in workflow_runs.items():
                total = len(runs)
                successful = sum(1 for r in runs if r.status == "success")
                latencies = [r.latency_ms for r in runs if r.latency_ms]
                costs = [r.total_cost for r in runs if hasattr(r, "total_cost")]

                metrics.append(
                    WorkflowMetrics(
                        workflow_name=name,
                        total_runs=total,
                        success_rate=successful / total if total > 0 else 0,
                        avg_latency_ms=np.mean(latencies) if latencies else 0,
                        p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
                        error_rate=1 - (successful / total) if total > 0 else 0,
                        cost_per_run=np.mean(costs) if costs else 0,
                    )
                )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get workflow metrics: {e}")
            return []

    async def get_agent_metrics(
        self, time_range: timedelta = timedelta(days=7)
    ) -> List[AgentMetrics]:
        """Get metrics for individual agents"""
        try:
            # Query agent runs
            runs = list(
                self.client.list_runs(
                    project_name=self.project_name,
                    filter={"start_time": datetime.now() - time_range, "tags": ["agent"]},
                    limit=1000,
                )
            )

            # Group by agent
            agent_runs = {}
            for run in runs:
                # Extract agent name from metadata
                agent_name = run.metadata.get("agent_type", "unknown")
                if agent_name not in agent_runs:
                    agent_runs[agent_name] = []
                agent_runs[agent_name].append(run)

            # Calculate metrics
            metrics = []
            for agent_name, runs in agent_runs.items():
                total = len(runs)
                successful = sum(1 for r in runs if r.status == "success")

                # Extract confidence scores from outputs
                confidences = []
                latencies = []
                for run in runs:
                    if run.outputs and isinstance(run.outputs, dict):
                        conf = run.outputs.get("confidence", 0)
                        confidences.append(conf)
                    if run.latency_ms:
                        latencies.append(run.latency_ms)

                metrics.append(
                    AgentMetrics(
                        agent_name=agent_name,
                        calls_count=total,
                        success_rate=successful / total if total > 0 else 0,
                        avg_confidence=np.mean(confidences) if confidences else 0,
                        avg_latency_ms=np.mean(latencies) if latencies else 0,
                    )
                )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get agent metrics: {e}")
            return []

    async def analyze_decision_patterns(
        self, symbol: Optional[str] = None, time_range: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """Analyze patterns in trading decisions"""
        try:
            # Query decision runs
            filters = {"start_time": datetime.now() - time_range, "tags": ["decision"]}

            if symbol:
                filters["tags"].append(symbol)

            runs = list(
                self.client.list_runs(project_name=self.project_name, filter=filters, limit=1000)
            )

            # Analyze patterns
            decisions = []
            for run in runs:
                if run.outputs:
                    decision = run.outputs.get("decision", {})
                    outcome = run.outputs.get("outcome", {})

                    decisions.append(
                        {
                            "timestamp": run.start_time,
                            "symbol": run.inputs.get("symbol", ""),
                            "action": decision.get("action", "HOLD"),
                            "confidence": decision.get("confidence", 0),
                            "profitable": outcome.get("profitable", None),
                            "return_pct": outcome.get("return_pct", 0),
                        }
                    )

            if not decisions:
                return {"error": "No decisions found"}

            # Convert to DataFrame for analysis
            df = pd.DataFrame(decisions)

            # Calculate patterns
            patterns = {
                "total_decisions": len(df),
                "action_distribution": df["action"].value_counts().to_dict(),
                "avg_confidence": df["confidence"].mean(),
                "confidence_by_action": df.groupby("action")["confidence"].mean().to_dict(),
                "profitability_rate": df["profitable"].mean() if "profitable" in df else None,
                "avg_return": df["return_pct"].mean() if "return_pct" in df else None,
                "best_performing_setups": self._find_best_setups(df),
                "worst_performing_setups": self._find_worst_setups(df),
            }

            return patterns

        except Exception as e:
            logger.error(f"Failed to analyze decision patterns: {e}")
            return {"error": str(e)}

    def _find_best_setups(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find best performing trading setups"""
        if "profitable" not in df.columns:
            return []

        # Group by confidence buckets and action
        df["confidence_bucket"] = pd.cut(df["confidence"], bins=[0, 0.6, 0.8, 1.0])

        best_setups = []
        for (action, conf_bucket), group in df.groupby(["action", "confidence_bucket"]):
            if len(group) >= 5:  # Minimum sample size
                profit_rate = group["profitable"].mean()
                avg_return = group["return_pct"].mean()

                if profit_rate > 0.6:  # 60% win rate
                    best_setups.append(
                        {
                            "action": action,
                            "confidence_range": str(conf_bucket),
                            "profit_rate": profit_rate,
                            "avg_return": avg_return,
                            "sample_size": len(group),
                        }
                    )

        return sorted(best_setups, key=lambda x: x["profit_rate"], reverse=True)[:5]

    def _find_worst_setups(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find worst performing trading setups"""
        if "profitable" not in df.columns:
            return []

        # Similar to best setups but inverted
        df["confidence_bucket"] = pd.cut(df["confidence"], bins=[0, 0.6, 0.8, 1.0])

        worst_setups = []
        for (action, conf_bucket), group in df.groupby(["action", "confidence_bucket"]):
            if len(group) >= 5:
                profit_rate = group["profitable"].mean()
                avg_return = group["return_pct"].mean()

                if profit_rate < 0.4:  # Less than 40% win rate
                    worst_setups.append(
                        {
                            "action": action,
                            "confidence_range": str(conf_bucket),
                            "profit_rate": profit_rate,
                            "avg_return": avg_return,
                            "sample_size": len(group),
                        }
                    )

        return sorted(worst_setups, key=lambda x: x["profit_rate"])[:5]

    def _register_default_evaluators(self):
        """Register default evaluation functions"""

        # Accuracy evaluator
        def accuracy_evaluator(run: Run) -> float:
            if not run.outputs or "decision" not in run.outputs:
                return 0.0

            decision = run.outputs["decision"]
            outcome = run.outputs.get("outcome", {})

            # Check if decision was correct
            if decision.get("action") == "BUY" and outcome.get("profitable", False):
                return 1.0
            elif decision.get("action") == "SELL" and not outcome.get("profitable", True):
                return 1.0
            elif decision.get("action") == "HOLD" and abs(outcome.get("return_pct", 0)) < 1:
                return 1.0

            return 0.0

        self.evaluators["accuracy"] = accuracy_evaluator

        # Risk-adjusted return evaluator
        def risk_adjusted_evaluator(run: Run) -> float:
            if not run.outputs:
                return 0.0

            outcome = run.outputs.get("outcome", {})
            risk = run.outputs.get("decision", {}).get("risk_score", 0.5)

            return_pct = outcome.get("return_pct", 0)
            # Sharpe-like ratio (simplified)
            risk_adjusted = return_pct / (risk + 0.1)

            # Normalize to 0-1
            return min(1.0, max(0.0, (risk_adjusted + 1) / 2))

        self.evaluators["risk_adjusted_return"] = risk_adjusted_evaluator

    def _track_workflow_success(self, workflow_name: str, result: Any):
        """Track workflow success metrics"""
        if workflow_name not in self.metrics_cache["workflows"]:
            self.metrics_cache["workflows"][workflow_name] = {"successes": 0, "total": 0}

        self.metrics_cache["workflows"][workflow_name]["total"] += 1
        if result and not isinstance(result, dict) or not result.get("error"):
            self.metrics_cache["workflows"][workflow_name]["successes"] += 1

    def _track_workflow_error(self, workflow_name: str, error: str):
        """Track workflow errors"""
        if workflow_name not in self.metrics_cache["workflows"]:
            self.metrics_cache["workflows"][workflow_name] = {
                "successes": 0,
                "total": 0,
                "errors": [],
            }

        self.metrics_cache["workflows"][workflow_name]["total"] += 1
        self.metrics_cache["workflows"][workflow_name]["errors"].append(
            {"timestamp": datetime.now().isoformat(), "error": error}
        )

    def _track_agent_performance(self, agent_name: str, result: Dict[str, Any], latency_ms: float):
        """Track agent performance metrics"""
        if agent_name not in self.metrics_cache["agents"]:
            self.metrics_cache["agents"][agent_name] = {
                "calls": 0,
                "total_confidence": 0,
                "total_latency": 0,
            }

        cache = self.metrics_cache["agents"][agent_name]
        cache["calls"] += 1
        cache["total_confidence"] += result.get("confidence", 0)
        cache["total_latency"] += latency_ms

    def _track_llm_usage(self, provider: str, model: str, usage: Any):
        """Track LLM token usage and costs"""
        # This would integrate with your cost tracking system
        logger.info(f"LLM usage - {provider}/{model}: {usage}")


# Singleton instance
observability = LangSmithObservability()


# Convenience decorators
trace_workflow = observability.trace_workflow
trace_agent = observability.trace_agent
trace_llm_call = observability.trace_llm_call


# High-level API functions
async def log_trading_decision(
    symbol: str,
    decision: Dict[str, Any],
    context: Dict[str, Any],
    outcome: Optional[Dict[str, Any]] = None,
):
    """Log a trading decision to LangSmith"""
    await observability.log_decision(symbol, decision, context, outcome)


async def get_system_metrics(time_range: timedelta = timedelta(days=7)) -> Dict[str, Any]:
    """Get comprehensive system metrics"""
    workflow_metrics = await observability.get_workflow_metrics(time_range=time_range)
    agent_metrics = await observability.get_agent_metrics(time_range=time_range)

    return {
        "workflows": [m.__dict__ for m in workflow_metrics],
        "agents": [m.__dict__ for m in agent_metrics],
        "timestamp": datetime.now().isoformat(),
    }


async def analyze_trading_performance(
    symbol: Optional[str] = None, days: int = 30
) -> Dict[str, Any]:
    """Analyze trading performance patterns"""
    return await observability.analyze_decision_patterns(
        symbol=symbol, time_range=timedelta(days=days)
    )
