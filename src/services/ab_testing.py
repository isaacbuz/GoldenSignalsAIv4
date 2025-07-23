"""
A/B Testing Framework for Agent Performance
Allows controlled experiments to test new agents against existing ones
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class ExperimentVariant:
    """Represents a variant in an A/B test"""
    name: str
    agent_config: Dict[str, Any]
    allocation_percentage: float
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    signal_count: int = 0
    start_time: Optional[datetime] = None
    
    def add_metric(self, metric_name: str, value: float):
        """Add a metric value to this variant"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        values = self.metrics.get(metric_name, [])
        if not values:
            return {"mean": 0, "std": 0, "count": 0, "min": 0, "max": 0}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "count": len(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values)
        }


@dataclass
class ABTestExperiment:
    """Represents an A/B test experiment"""
    experiment_id: str
    name: str
    description: str
    control_variant: ExperimentVariant
    test_variants: List[ExperimentVariant]
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    min_sample_size: int = 100
    confidence_level: float = 0.95
    target_metrics: List[str] = field(default_factory=list)
    
    def get_all_variants(self) -> List[ExperimentVariant]:
        """Get all variants including control"""
        return [self.control_variant] + self.test_variants
    
    def is_active(self) -> bool:
        """Check if experiment is currently running"""
        return self.status == ExperimentStatus.RUNNING


class ABTestingService:
    """Service for managing A/B tests for trading agents"""
    
    def __init__(self, redis_client=None):
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.redis_client = redis_client
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments from Redis if available"""
        if self.redis_client:
            try:
                keys = self.redis_client.keys("ab_test:*")
                for key in keys:
                    data = self.redis_client.get(key)
                    if data:
                        exp_data = json.loads(data)
                        # Reconstruct experiment from stored data
                        # This is simplified - in production you'd have proper serialization
                        pass
            except Exception as e:
                logger.error(f"Failed to load experiments: {e}")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        control_agent_config: Dict[str, Any],
        test_agent_configs: List[Dict[str, Any]],
        allocation_percentages: Optional[List[float]] = None,
        target_metrics: Optional[List[str]] = None,
        min_sample_size: int = 100
    ) -> ABTestExperiment:
        """
        Create a new A/B test experiment
        
        Args:
            name: Experiment name
            description: Detailed description
            control_agent_config: Configuration for control agent
            test_agent_configs: List of test agent configurations
            allocation_percentages: Traffic allocation (must sum to 1.0)
            target_metrics: Metrics to track
            min_sample_size: Minimum samples before analysis
            
        Returns:
            Created experiment
        """
        experiment_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Default equal allocation if not specified
        if allocation_percentages is None:
            n_variants = len(test_agent_configs) + 1
            allocation_percentages = [1.0 / n_variants] * n_variants
        else:
            # Validate allocations sum to 1.0
            total = sum(allocation_percentages)
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Allocations must sum to 1.0, got {total}")
        
        # Create control variant
        control_variant = ExperimentVariant(
            name="control",
            agent_config=control_agent_config,
            allocation_percentage=allocation_percentages[0]
        )
        
        # Create test variants
        test_variants = []
        for i, (config, alloc) in enumerate(zip(test_agent_configs, allocation_percentages[1:])):
            variant = ExperimentVariant(
                name=f"variant_{i+1}",
                agent_config=config,
                allocation_percentage=alloc
            )
            test_variants.append(variant)
        
        # Default target metrics
        if target_metrics is None:
            target_metrics = ["accuracy", "profit_loss", "sharpe_ratio", "win_rate"]
        
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            control_variant=control_variant,
            test_variants=test_variants,
            status=ExperimentStatus.DRAFT,
            created_at=datetime.now(),
            min_sample_size=min_sample_size,
            target_metrics=target_metrics
        )
        
        self.experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created A/B test experiment: {experiment_id}")
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Experiment {experiment_id} is not in DRAFT status")
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        
        # Initialize variant start times
        for variant in experiment.get_all_variants():
            variant.start_time = datetime.now()
        
        self._save_experiment(experiment)
        logger.info(f"Started experiment: {experiment_id}")
        return True
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        symbol: str
    ) -> Optional[ExperimentVariant]:
        """
        Assign a user to a variant for a specific symbol
        
        Args:
            experiment_id: The experiment ID
            user_id: User identifier
            symbol: Trading symbol
            
        Returns:
            Assigned variant or None if experiment not active
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active():
            return None
        
        # Create deterministic assignment based on user + symbol
        assignment_key = f"{experiment_id}:{user_id}:{symbol}"
        assignment_hash = int(hashlib.md5(assignment_key.encode()).hexdigest(), 16)
        assignment_value = (assignment_hash % 100) / 100.0
        
        # Assign to variant based on allocation
        cumulative = 0.0
        for variant in experiment.get_all_variants():
            cumulative += variant.allocation_percentage
            if assignment_value < cumulative:
                return variant
        
        # Fallback to control
        return experiment.control_variant
    
    def record_signal_result(
        self,
        experiment_id: str,
        variant_name: str,
        metrics: Dict[str, float]
    ):
        """
        Record the result of a signal for a variant
        
        Args:
            experiment_id: The experiment ID
            variant_name: Name of the variant
            metrics: Dict of metric name to value
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return
        
        # Find variant
        variant = None
        for v in experiment.get_all_variants():
            if v.name == variant_name:
                variant = v
                break
        
        if not variant:
            logger.error(f"Variant {variant_name} not found in experiment {experiment_id}")
            return
        
        # Record metrics
        for metric_name, value in metrics.items():
            if metric_name in experiment.target_metrics:
                variant.add_metric(metric_name, value)
        
        variant.signal_count += 1
        self._save_experiment(experiment)
    
    def analyze_experiment(
        self,
        experiment_id: str,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Analyze experiment results for a specific metric
        
        Args:
            experiment_id: The experiment ID
            metric_name: Metric to analyze
            
        Returns:
            Analysis results including statistical significance
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        results = {
            "experiment_id": experiment_id,
            "metric": metric_name,
            "status": experiment.status.value,
            "variants": {}
        }
        
        # Get control stats
        control_stats = experiment.control_variant.get_metric_stats(metric_name)
        results["variants"]["control"] = control_stats
        
        # Analyze each test variant
        for variant in experiment.test_variants:
            variant_stats = variant.get_metric_stats(metric_name)
            results["variants"][variant.name] = variant_stats
            
            # Perform statistical test if enough samples
            if (control_stats["count"] >= experiment.min_sample_size and
                variant_stats["count"] >= experiment.min_sample_size):
                
                # Perform t-test
                control_values = experiment.control_variant.metrics.get(metric_name, [])
                variant_values = variant.metrics.get(metric_name, [])
                
                if control_values and variant_values:
                    t_stat, p_value = stats.ttest_ind(control_values, variant_values)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (np.std(control_values)**2 + np.std(variant_values)**2) / 2
                    )
                    effect_size = (variant_stats["mean"] - control_stats["mean"]) / pooled_std if pooled_std > 0 else 0
                    
                    # Determine if significant
                    is_significant = p_value < (1 - experiment.confidence_level)
                    
                    # Calculate lift
                    lift = ((variant_stats["mean"] - control_stats["mean"]) / 
                           control_stats["mean"] * 100) if control_stats["mean"] != 0 else 0
                    
                    results["variants"][variant.name].update({
                        "p_value": p_value,
                        "t_statistic": t_stat,
                        "effect_size": effect_size,
                        "is_significant": is_significant,
                        "lift_percentage": lift,
                        "confidence_interval": self._calculate_confidence_interval(
                            variant_values,
                            experiment.confidence_level
                        )
                    })
        
        # Add recommendation
        results["recommendation"] = self._generate_recommendation(results, experiment)
        
        return results
    
    def _calculate_confidence_interval(
        self,
        values: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a set of values"""
        if not values:
            return (0, 0)
        
        mean = np.mean(values)
        std_err = stats.sem(values)
        interval = std_err * stats.t.ppf((1 + confidence_level) / 2, len(values) - 1)
        
        return (mean - interval, mean + interval)
    
    def _generate_recommendation(
        self,
        results: Dict[str, Any],
        experiment: ABTestExperiment
    ) -> Dict[str, str]:
        """Generate recommendation based on analysis results"""
        recommendation = {
            "action": "continue",
            "reasoning": ""
        }
        
        # Check if any variant is significantly better
        best_variant = None
        best_lift = 0
        
        for variant_name, variant_data in results["variants"].items():
            if variant_name == "control":
                continue
                
            if variant_data.get("is_significant", False) and variant_data.get("lift_percentage", 0) > best_lift:
                best_variant = variant_name
                best_lift = variant_data["lift_percentage"]
        
        if best_variant and best_lift > 5:  # 5% improvement threshold
            recommendation["action"] = "deploy_winner"
            recommendation["reasoning"] = (
                f"{best_variant} shows {best_lift:.1f}% improvement "
                f"with statistical significance"
            )
            recommendation["winner"] = best_variant
        elif all(v.signal_count >= experiment.min_sample_size * 2 
                for v in experiment.get_all_variants()):
            recommendation["action"] = "stop"
            recommendation["reasoning"] = "No significant improvement found after sufficient samples"
        else:
            recommendation["action"] = "continue"
            recommendation["reasoning"] = "Need more data for conclusive results"
        
        return recommendation
    
    def stop_experiment(self, experiment_id: str, reason: str = ""):
        """Stop an experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.now()
        self._save_experiment(experiment)
        
        logger.info(f"Stopped experiment {experiment_id}: {reason}")
    
    def _save_experiment(self, experiment: ABTestExperiment):
        """Save experiment to Redis if available"""
        if self.redis_client:
            try:
                # Simplified serialization - in production use proper serialization
                key = f"ab_test:{experiment.experiment_id}"
                # self.redis_client.set(key, json.dumps(experiment_data))
                pass
            except Exception as e:
                logger.error(f"Failed to save experiment: {e}")
    
    def get_active_experiments(self) -> List[ABTestExperiment]:
        """Get all active experiments"""
        return [exp for exp in self.experiments.values() 
                if exp.is_active()]
    
    def get_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for an experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        report = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "ended_at": experiment.ended_at.isoformat() if experiment.ended_at else None,
            "duration_days": None,
            "variants": [],
            "metrics_analysis": {}
        }
        
        # Calculate duration
        if experiment.started_at:
            end_time = experiment.ended_at or datetime.now()
            duration = end_time - experiment.started_at
            report["duration_days"] = duration.days
        
        # Variant summaries
        for variant in experiment.get_all_variants():
            variant_summary = {
                "name": variant.name,
                "allocation": variant.allocation_percentage,
                "signal_count": variant.signal_count,
                "metrics": {}
            }
            
            for metric in experiment.target_metrics:
                variant_summary["metrics"][metric] = variant.get_metric_stats(metric)
            
            report["variants"].append(variant_summary)
        
        # Analyze each metric
        for metric in experiment.target_metrics:
            report["metrics_analysis"][metric] = self.analyze_experiment(
                experiment_id, metric
            )
        
        return report


# Global A/B testing service instance
_ab_testing_service: Optional[ABTestingService] = None


def get_ab_testing_service(redis_client=None) -> ABTestingService:
    """Get or create A/B testing service"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService(redis_client)
    return _ab_testing_service