#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI - Monitoring Service
Comprehensive monitoring for all system metrics

Features:
- Latency tracking
- Cache hit rates
- API usage monitoring
- Performance metrics
- Alert thresholds
- Real-time dashboards
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collectio, timezonens import deque, defaultdict
import statistics
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str] = None

@dataclass
class Alert:
    """Alert configuration"""
    name: str
    metric: str
    threshold: float
    condition: str  # "above", "below"
    duration: int  # seconds
    enabled: bool = True

class MetricType:
    """Metric types"""
    LATENCY = "latency"
    CACHE_HIT_RATE = "cache_hit_rate"
    API_CALLS = "api_calls"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUEUE_SIZE = "queue_size"
    MEMORY_USAGE = "memory_usage"
    CONNECTION_COUNT = "connection_count"

class MonitoringService:
    """Comprehensive monitoring service"""
    
    def __init__(self):
        # Metrics storage (last 1 hour of data)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))
        
        # Aggregated stats
        self.stats = {
            "latency": {
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "avg": 0,
                "min": 0,
                "max": 0
            },
            "cache": {
                "hit_rate": 0,
                "total_hits": 0,
                "total_misses": 0
            },
            "api": {
                "total_calls": 0,
                "calls_per_minute": 0,
                "by_endpoint": defaultdict(int),
                "by_source": defaultdict(int)
            },
            "errors": {
                "total": 0,
                "rate": 0,
                "by_type": defaultdict(int)
            }
        }
        
        # Alerts configuration
        self.alerts = [
            Alert("High Latency", MetricType.LATENCY, 1000, "above", 60),
            Alert("Low Cache Hit Rate", MetricType.CACHE_HIT_RATE, 0.5, "below", 300),
            Alert("High Error Rate", MetricType.ERROR_RATE, 0.05, "above", 60),
            Alert("API Rate Limit", MetricType.API_CALLS, 50, "above", 60)
        ]
        
        # Alert state
        self.alert_state = {}
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
    
    def _ensure_background_tasks(self):
        """Ensure background tasks are started when in async context"""
        if not self._background_tasks_started:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._aggregation_task())
                loop.create_task(self._alert_task())
                self._background_tasks_started = True
            except RuntimeError:
                # No event loop running yet, tasks will start when one is available
                pass
    
    def record_latency(self, operation: str, latency_ms: float, tags: Dict[str, str] = None):
        """Record latency metric"""
        # Ensure background tasks are running
        self._ensure_background_tasks()
        
        tags = tags or {}
        tags["operation"] = operation
        
        metric = MetricPoint(
            timestamp=time.time(),
            value=latency_ms,
            tags=tags
        )
        
        self.metrics[f"{MetricType.LATENCY}:{operation}"].append(metric)
        self.metrics[MetricType.LATENCY].append(metric)
    
    def record_cache_hit(self, cache_level: str, hit: bool):
        """Record cache hit/miss"""
        metric = MetricPoint(
            timestamp=time.time(),
            value=1 if hit else 0,
            tags={"level": cache_level}
        )
        
        self.metrics[MetricType.CACHE_HIT_RATE].append(metric)
        
        if hit:
            self.stats["cache"]["total_hits"] += 1
        else:
            self.stats["cache"]["total_misses"] += 1
    
    def record_api_call(self, endpoint: str, source: str, success: bool = True):
        """Record API call"""
        self.stats["api"]["total_calls"] += 1
        self.stats["api"]["by_endpoint"][endpoint] += 1
        self.stats["api"]["by_source"][source] += 1
        
        metric = MetricPoint(
            timestamp=time.time(),
            value=1,
            tags={"endpoint": endpoint, "source": source, "success": str(success)}
        )
        
        self.metrics[MetricType.API_CALLS].append(metric)
        
        if not success:
            self.record_error("api_error", {"endpoint": endpoint})
    
    def record_error(self, error_type: str, tags: Dict[str, str] = None):
        """Record error"""
        self.stats["errors"]["total"] += 1
        self.stats["errors"]["by_type"][error_type] += 1
        
        metric = MetricPoint(
            timestamp=time.time(),
            value=1,
            tags={"type": error_type, **(tags or {})}
        )
        
        self.metrics[MetricType.ERROR_RATE].append(metric)
    
    def get_latency_stats(self, operation: str = None, window_seconds: int = 300) -> Dict[str, float]:
        """Get latency statistics"""
        metric_key = f"{MetricType.LATENCY}:{operation}" if operation else MetricType.LATENCY
        metrics = list(self.metrics[metric_key])
        
        if not metrics:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}
        
        # Filter by time window
        cutoff = time.time() - window_seconds
        values = [m.value for m in metrics if m.timestamp > cutoff]
        
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}
        
        values.sort()
        
        return {
            "p50": values[int(len(values) * 0.5)],
            "p95": values[int(len(values) * 0.95)],
            "p99": values[int(len(values) * 0.99)],
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
    
    def get_cache_hit_rate(self, window_seconds: int = 300) -> float:
        """Get cache hit rate"""
        metrics = list(self.metrics[MetricType.CACHE_HIT_RATE])
        
        if not metrics:
            return 0.0
        
        cutoff = time.time() - window_seconds
        values = [m.value for m in metrics if m.timestamp > cutoff]
        
        if not values:
            return 0.0
        
        return sum(values) / len(values)
    
    def get_api_rate(self, window_seconds: int = 60) -> float:
        """Get API calls per second"""
        metrics = list(self.metrics[MetricType.API_CALLS])
        
        if not metrics:
            return 0.0
        
        cutoff = time.time() - window_seconds
        count = sum(1 for m in metrics if m.timestamp > cutoff)
        
        return count / window_seconds
    
    def get_error_rate(self, window_seconds: int = 300) -> float:
        """Get error rate"""
        error_metrics = list(self.metrics[MetricType.ERROR_RATE])
        api_metrics = list(self.metrics[MetricType.API_CALLS])
        
        cutoff = time.time() - window_seconds
        
        errors = sum(1 for m in error_metrics if m.timestamp > cutoff)
        total = sum(1 for m in api_metrics if m.timestamp > cutoff)
        
        if total == 0:
            return 0.0
        
        return errors / total
    
    async def _aggregation_task(self):
        """Aggregate metrics periodically"""
        while True:
            try:
                # Update latency stats
                self.stats["latency"] = self.get_latency_stats()
                
                # Update cache hit rate
                self.stats["cache"]["hit_rate"] = self.get_cache_hit_rate()
                
                # Update API rate
                self.stats["api"]["calls_per_minute"] = self.get_api_rate() * 60
                
                # Update error rate
                self.stats["errors"]["rate"] = self.get_error_rate()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                await asyncio.sleep(10)
    
    async def _alert_task(self):
        """Check alerts periodically"""
        while True:
            try:
                for alert in self.alerts:
                    if not alert.enabled:
                        continue
                    
                    # Get current metric value
                    if alert.metric == MetricType.LATENCY:
                        value = self.stats["latency"]["p95"]
                    elif alert.metric == MetricType.CACHE_HIT_RATE:
                        value = self.stats["cache"]["hit_rate"]
                    elif alert.metric == MetricType.ERROR_RATE:
                        value = self.stats["errors"]["rate"]
                    elif alert.metric == MetricType.API_CALLS:
                        value = self.stats["api"]["calls_per_minute"]
                    else:
                        continue
                    
                    # Check threshold
                    triggered = False
                    if alert.condition == "above" and value > alert.threshold:
                        triggered = True
                    elif alert.condition == "below" and value < alert.threshold:
                        triggered = True
                    
                    # Update alert state
                    if triggered:
                        if alert.name not in self.alert_state:
                            self.alert_state[alert.name] = time.time()
                        elif time.time() - self.alert_state[alert.name] > alert.duration:
                            # Alert triggered
                            logger.warning(f"🚨 Alert: {alert.name} - {alert.metric}={value:.2f}")
                            # In production, send notification
                    else:
                        # Clear alert state
                        if alert.name in self.alert_state:
                            del self.alert_state[alert.name]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Alert check error: {e}")
                await asyncio.sleep(5)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency": self.stats["latency"],
            "cache": self.stats["cache"],
            "api": {
                "total_calls": self.stats["api"]["total_calls"],
                "calls_per_minute": self.stats["api"]["calls_per_minute"],
                "top_endpoints": dict(sorted(
                    self.stats["api"]["by_endpoint"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            "errors": {
                "total": self.stats["errors"]["total"],
                "rate": f"{self.stats['errors']['rate'] * 100:.2f}%",
                "top_errors": dict(sorted(
                    self.stats["errors"]["by_type"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            },
            "alerts": [
                {
                    "name": alert.name,
                    "active": alert.name in self.alert_state,
                    "duration": time.time() - self.alert_state.get(alert.name, 0)
                }
                for alert in self.alerts
            ]
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics for external monitoring systems"""
        if format == "prometheus":
            # Prometheus format
            lines = []
            
            # Latency metrics
            for key, value in self.stats["latency"].items():
                lines.append(f'latency_{key} {value}')
            
            # Cache metrics
            lines.append(f'cache_hit_rate {self.stats["cache"]["hit_rate"]}')
            lines.append(f'cache_total_hits {self.stats["cache"]["total_hits"]}')
            
            # API metrics
            lines.append(f'api_calls_total {self.stats["api"]["total_calls"]}')
            lines.append(f'api_calls_per_minute {self.stats["api"]["calls_per_minute"]}')
            
            # Error metrics
            lines.append(f'error_rate {self.stats["errors"]["rate"]}')
            lines.append(f'errors_total {self.stats["errors"]["total"]}')
            
            return "\n".join(lines)
        
        else:
            # JSON format
            return json.dumps(self.get_dashboard_data(), indent=2)

# Singleton instance
_monitoring_service: Optional[MonitoringService] = None

def get_monitoring_service() -> MonitoringService:
    """Get or create monitoring service singleton"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service 