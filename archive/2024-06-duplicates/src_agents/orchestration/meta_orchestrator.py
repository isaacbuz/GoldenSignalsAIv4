"""
Meta Orchestrator for GoldenSignalsAI
Higher-level orchestration across multiple orchestrators and strategies
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import json

from agents.core.unified_base_agent import UnifiedBaseAgent, AgentType, MessagePriority, AgentMessage
from agents.orchestration.agent_orchestrator import (
    AgentOrchestrator, OrchestrationStrategy, WorkflowDefinition, AgentTask
)


class MetaStrategy(Enum):
    """Meta-level orchestration strategies"""
    MARKET_ADAPTIVE = "market_adaptive"  # Adapt to market conditions
    RISK_AWARE = "risk_aware"  # Prioritize risk management
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Optimize for best performance
    MULTI_TIMEFRAME = "multi_timeframe"  # Coordinate across timeframes
    SENTIMENT_DRIVEN = "sentiment_driven"  # Let sentiment guide strategy


@dataclass
class MarketRegime:
    """Current market regime classification"""
    regime_type: str  # bull, bear, sideways, volatile
    confidence: float
    volatility: float
    trend_strength: float
    sentiment_score: float
    timestamp: datetime


@dataclass
class MetaWorkflow:
    """High-level workflow spanning multiple orchestrators"""
    workflow_id: str
    name: str
    meta_strategy: MetaStrategy
    sub_workflows: List[WorkflowDefinition]
    market_context: Dict[str, Any]
    risk_parameters: Dict[str, float]
    performance_targets: Dict[str, float]
    coordination_rules: List[Dict[str, Any]]


class MetaOrchestrator(UnifiedBaseAgent):
    """Meta-level orchestrator for complex multi-strategy coordination"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.ORCHESTRATOR, config)
        
        # Sub-orchestrators
        self.orchestrators: Dict[str, AgentOrchestrator] = {}
        self.orchestrator_specializations: Dict[str, List[str]] = {
            'technical': ['momentum', 'mean_reversion', 'pattern_recognition'],
            'fundamental': ['value', 'growth', 'quality'],
            'sentiment': ['news', 'social', 'market_psychology'],
            'quantitative': ['statistical_arbitrage', 'factor_models', 'ml_predictions'],
            'risk': ['portfolio_risk', 'position_sizing', 'hedging']
        }
        
        # Market regime detection
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[MarketRegime] = []
        self.regime_transition_rules: Dict[str, Dict[str, Any]] = {}
        
        # Strategy selection
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0
        })
        self.strategy_weights: Dict[str, float] = {}
        
        # Coordination state
        self.active_meta_workflows: Dict[str, MetaWorkflow] = {}
        self.coordination_locks: Dict[str, asyncio.Lock] = {}
        
        # Performance optimization
        self.optimization_history: List[Dict[str, Any]] = []
        self.adaptation_rate: float = config.get('adaptation_rate', 0.1)
        
    def _register_capabilities(self):
        """Register meta orchestrator capabilities"""
        self.capabilities = {
            'execute_meta_workflow': {
                'description': 'Execute high-level multi-strategy workflow',
                'input': {'meta_workflow': 'MetaWorkflow'},
                'output': {'results': 'Dict[str, Any]'}
            },
            'detect_market_regime': {
                'description': 'Detect current market regime',
                'input': {'market_data': 'Dict[str, Any]'},
                'output': {'regime': 'MarketRegime'}
            },
            'optimize_strategy_allocation': {
                'description': 'Optimize strategy weights',
                'input': {'performance_data': 'Dict[str, Any]'},
                'output': {'weights': 'Dict[str, float]'}
            },
            'coordinate_timeframes': {
                'description': 'Coordinate strategies across timeframes',
                'input': {'timeframes': 'List[str]'},
                'output': {'coordination_plan': 'Dict[str, Any]'}
            }
        }
    
    def _register_message_handlers(self):
        """Register message handlers"""
        self.message_handlers = {
            'execute_meta_workflow': self.handle_execute_meta_workflow,
            'update_market_regime': self.handle_update_market_regime,
            'strategy_performance_update': self.handle_strategy_performance_update,
            'coordinate_strategies': self.handle_coordinate_strategies,
            'emergency_risk_override': self.handle_emergency_risk_override
        }
    
    async def handle_execute_meta_workflow(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a meta-level workflow"""
        meta_workflow = MetaWorkflow(**message.payload['meta_workflow'])
        
        # Store workflow
        self.active_meta_workflows[meta_workflow.workflow_id] = meta_workflow
        
        try:
            # Detect current market regime
            regime = await self._detect_market_regime(meta_workflow.market_context)
            
            # Select appropriate strategies based on regime and meta-strategy
            selected_strategies = await self._select_strategies(
                meta_workflow.meta_strategy,
                regime,
                meta_workflow.risk_parameters
            )
            
            # Create sub-workflows for each selected strategy
            sub_workflows = await self._create_sub_workflows(
                selected_strategies,
                meta_workflow
            )
            
            # Execute sub-workflows with coordination
            results = await self._execute_coordinated_workflows(
                sub_workflows,
                meta_workflow.coordination_rules
            )
            
            # Aggregate and optimize results
            final_results = await self._aggregate_results(
                results,
                meta_workflow.performance_targets
            )
            
            # Update strategy performance
            await self._update_strategy_performance(selected_strategies, final_results)
            
            return {
                'workflow_id': meta_workflow.workflow_id,
                'meta_strategy': meta_workflow.meta_strategy.value,
                'regime': regime.__dict__,
                'strategies_used': selected_strategies,
                'results': final_results,
                'performance_metrics': self._calculate_performance_metrics(final_results)
            }
            
        except Exception as e:
            self.logger.error(f"Meta workflow execution error: {e}")
            return {'error': str(e)}
        finally:
            # Clean up
            del self.active_meta_workflows[meta_workflow.workflow_id]
    
    async def _detect_market_regime(self, market_context: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime"""
        # Extract market indicators
        volatility = market_context.get('volatility', 0.0)
        trend = market_context.get('trend_strength', 0.0)
        volume = market_context.get('volume_ratio', 1.0)
        sentiment = market_context.get('sentiment_score', 0.0)
        
        # Classify regime
        if volatility > 0.3:
            regime_type = 'volatile'
        elif trend > 0.5:
            regime_type = 'bull'
        elif trend < -0.5:
            regime_type = 'bear'
        else:
            regime_type = 'sideways'
        
        # Calculate confidence based on indicator agreement
        indicators = [
            abs(volatility) > 0.2,
            abs(trend) > 0.3,
            volume > 1.2 or volume < 0.8,
            abs(sentiment) > 0.3
        ]
        confidence = sum(indicators) / len(indicators)
        
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend,
            sentiment_score=sentiment,
            timestamp=datetime.now()
        )
        
        # Update regime tracking
        self.current_regime = regime
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
        
        return regime
    
    async def _select_strategies(self, 
                               meta_strategy: MetaStrategy,
                               regime: MarketRegime,
                               risk_params: Dict[str, float]) -> List[str]:
        """Select strategies based on meta-strategy and market regime"""
        selected = []
        
        if meta_strategy == MetaStrategy.MARKET_ADAPTIVE:
            # Adapt to current market regime
            if regime.regime_type == 'bull':
                selected.extend(['momentum', 'growth', 'sentiment_long'])
            elif regime.regime_type == 'bear':
                selected.extend(['defensive', 'value', 'hedging'])
            elif regime.regime_type == 'volatile':
                selected.extend(['mean_reversion', 'volatility_arbitrage', 'options'])
            else:  # sideways
                selected.extend(['range_trading', 'pairs_trading', 'theta_harvesting'])
                
        elif meta_strategy == MetaStrategy.RISK_AWARE:
            # Prioritize risk management
            max_risk = risk_params.get('max_portfolio_risk', 0.02)
            if max_risk < 0.01:
                selected.extend(['market_neutral', 'arbitrage', 'fixed_income'])
            else:
                selected.extend(['balanced', 'risk_parity', 'tail_hedging'])
                
        elif meta_strategy == MetaStrategy.PERFORMANCE_OPTIMIZED:
            # Select best performing strategies
            sorted_strategies = sorted(
                self.strategy_performance.items(),
                key=lambda x: x[1]['sharpe_ratio'],
                reverse=True
            )
            selected.extend([s[0] for s in sorted_strategies[:5]])
            
        elif meta_strategy == MetaStrategy.MULTI_TIMEFRAME:
            # Coordinate across timeframes
            selected.extend([
                'intraday_momentum',
                'daily_swing',
                'weekly_trend',
                'monthly_positioning'
            ])
            
        elif meta_strategy == MetaStrategy.SENTIMENT_DRIVEN:
            # Let sentiment guide selection
            if regime.sentiment_score > 0.5:
                selected.extend(['sentiment_momentum', 'social_alpha', 'news_driven'])
            else:
                selected.extend(['contrarian', 'fear_gauge', 'sentiment_reversal'])
        
        # Apply risk filters
        filtered = []
        for strategy in selected:
            strategy_risk = self.strategy_performance[strategy].get('max_drawdown', 0)
            if abs(strategy_risk) <= risk_params.get('max_strategy_drawdown', 0.1):
                filtered.append(strategy)
        
        return filtered if filtered else selected[:3]  # Fallback to top 3
    
    async def _create_sub_workflows(self,
                                  strategies: List[str],
                                  meta_workflow: MetaWorkflow) -> List[WorkflowDefinition]:
        """Create sub-workflows for selected strategies"""
        sub_workflows = []
        
        for strategy in strategies:
            # Get appropriate orchestrator for strategy
            orchestrator_type = self._get_orchestrator_for_strategy(strategy)
            
            # Create workflow definition
            workflow = WorkflowDefinition(
                workflow_id=f"{meta_workflow.workflow_id}_{strategy}",
                name=f"{strategy}_workflow",
                tasks=self._create_strategy_tasks(strategy, meta_workflow),
                strategy=self._get_orchestration_strategy(strategy),
                timeout=meta_workflow.market_context.get('execution_timeout', 60.0),
                metadata={
                    'parent_workflow': meta_workflow.workflow_id,
                    'strategy': strategy,
                    'risk_parameters': meta_workflow.risk_parameters
                }
            )
            
            sub_workflows.append(workflow)
        
        return sub_workflows
    
    def _get_orchestrator_for_strategy(self, strategy: str) -> str:
        """Determine which orchestrator handles a strategy"""
        for orch_type, strategies in self.orchestrator_specializations.items():
            if strategy in strategies:
                return orch_type
        return 'general'
    
    def _get_orchestration_strategy(self, strategy: str) -> OrchestrationStrategy:
        """Get orchestration strategy for a specific trading strategy"""
        strategy_mapping = {
            'momentum': OrchestrationStrategy.SEQUENTIAL,
            'mean_reversion': OrchestrationStrategy.PARALLEL,
            'sentiment_momentum': OrchestrationStrategy.ENSEMBLE,
            'multi_factor': OrchestrationStrategy.HIERARCHICAL,
            'adaptive_trading': OrchestrationStrategy.ADAPTIVE
        }
        return strategy_mapping.get(strategy, OrchestrationStrategy.PARALLEL)
    
    def _create_strategy_tasks(self, 
                             strategy: str,
                             meta_workflow: MetaWorkflow) -> List[AgentTask]:
        """Create tasks for a specific strategy"""
        tasks = []
        
        # Common analysis tasks
        tasks.append(AgentTask(
            task_id=f"{strategy}_market_analysis",
            agent_id="market_analyzer",
            task_type="analyze_market",
            params={
                'symbols': meta_workflow.market_context.get('symbols', ['SPY']),
                'timeframe': meta_workflow.market_context.get('timeframe', '5m'),
                'strategy': strategy
            },
            priority=MessagePriority.HIGH
        ))
        
        # Strategy-specific tasks
        if strategy == 'momentum':
            tasks.extend(self._create_momentum_tasks(meta_workflow))
        elif strategy == 'mean_reversion':
            tasks.extend(self._create_mean_reversion_tasks(meta_workflow))
        elif strategy == 'sentiment_momentum':
            tasks.extend(self._create_sentiment_tasks(meta_workflow))
        # Add more strategy-specific task creators...
        
        return tasks
    
    def _create_momentum_tasks(self, meta_workflow: MetaWorkflow) -> List[AgentTask]:
        """Create momentum strategy tasks"""
        return [
            AgentTask(
                task_id="momentum_screening",
                agent_id="momentum_agent",
                task_type="screen_momentum",
                params={
                    'min_rsi': 60,
                    'min_volume_ratio': 1.5,
                    'lookback_period': 20
                },
                dependencies=["momentum_market_analysis"]
            ),
            AgentTask(
                task_id="momentum_signals",
                agent_id="momentum_agent",
                task_type="generate_signals",
                params={
                    'strategy': 'momentum_breakout',
                    'risk_per_trade': meta_workflow.risk_parameters.get('risk_per_trade', 0.01)
                },
                dependencies=["momentum_screening"]
            )
        ]
    
    def _create_mean_reversion_tasks(self, meta_workflow: MetaWorkflow) -> List[AgentTask]:
        """Create mean reversion strategy tasks"""
        return [
            AgentTask(
                task_id="mean_reversion_analysis",
                agent_id="mean_reversion_agent",
                task_type="identify_extremes",
                params={
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'rsi_period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                },
                dependencies=["mean_reversion_market_analysis"]
            ),
            AgentTask(
                task_id="mean_reversion_signals",
                agent_id="mean_reversion_agent",
                task_type="generate_signals",
                params={
                    'strategy': 'bollinger_reversal',
                    'position_size': meta_workflow.risk_parameters.get('position_size', 0.1)
                },
                dependencies=["mean_reversion_analysis"]
            )
        ]
    
    def _create_sentiment_tasks(self, meta_workflow: MetaWorkflow) -> List[AgentTask]:
        """Create sentiment strategy tasks"""
        return [
            AgentTask(
                task_id="sentiment_analysis",
                agent_id="sentiment_agent",
                task_type="analyze_sentiment",
                params={
                    'sources': ['news', 'social', 'options_flow'],
                    'lookback_hours': 24,
                    'min_confidence': 0.7
                },
                dependencies=["sentiment_momentum_market_analysis"]
            ),
            AgentTask(
                task_id="sentiment_signals",
                agent_id="sentiment_agent",
                task_type="generate_signals",
                params={
                    'strategy': 'sentiment_momentum',
                    'sentiment_threshold': 0.6
                },
                dependencies=["sentiment_analysis"]
            )
        ]
    
    async def _execute_coordinated_workflows(self,
                                           workflows: List[WorkflowDefinition],
                                           coordination_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute workflows with coordination"""
        results = {}
        
        # Group workflows by coordination requirements
        independent = []
        coordinated = defaultdict(list)
        
        for workflow in workflows:
            coord_group = self._get_coordination_group(workflow, coordination_rules)
            if coord_group:
                coordinated[coord_group].append(workflow)
            else:
                independent.append(workflow)
        
        # Execute independent workflows in parallel
        if independent:
            independent_tasks = []
            for workflow in independent:
                task = self._execute_single_workflow(workflow)
                independent_tasks.append(task)
            
            independent_results = await asyncio.gather(*independent_tasks)
            for workflow, result in zip(independent, independent_results):
                results[workflow.workflow_id] = result
        
        # Execute coordinated groups
        for group_name, group_workflows in coordinated.items():
            group_results = await self._execute_coordinated_group(
                group_workflows,
                coordination_rules
            )
            results.update(group_results)
        
        return results
    
    def _get_coordination_group(self, 
                               workflow: WorkflowDefinition,
                               coordination_rules: List[Dict[str, Any]]) -> Optional[str]:
        """Determine coordination group for workflow"""
        for rule in coordination_rules:
            if self._matches_rule(workflow, rule):
                return rule.get('group_name')
        return None
    
    def _matches_rule(self, workflow: WorkflowDefinition, rule: Dict[str, Any]) -> bool:
        """Check if workflow matches coordination rule"""
        # Check strategy match
        if 'strategies' in rule:
            workflow_strategy = workflow.metadata.get('strategy')
            if workflow_strategy not in rule['strategies']:
                return False
        
        # Check other conditions
        if 'condition' in rule:
            # Evaluate condition (simplified)
            return True
        
        return True
    
    async def _execute_coordinated_group(self,
                                       workflows: List[WorkflowDefinition],
                                       coordination_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a coordinated group of workflows"""
        results = {}
        
        # Get coordination lock for group
        group_id = f"coord_{workflows[0].workflow_id}"
        if group_id not in self.coordination_locks:
            self.coordination_locks[group_id] = asyncio.Lock()
        
        async with self.coordination_locks[group_id]:
            # Execute workflows with coordination
            for i, workflow in enumerate(workflows):
                # Check if we should wait for previous results
                if i > 0:
                    await asyncio.sleep(0.1)  # Small delay for coordination
                
                result = await self._execute_single_workflow(workflow)
                results[workflow.workflow_id] = result
                
                # Share results with other workflows if needed
                for rule in coordination_rules:
                    if rule.get('share_results'):
                        # Update remaining workflows with results
                        for j in range(i + 1, len(workflows)):
                            workflows[j].metadata['shared_results'] = result
        
        return results
    
    async def _execute_single_workflow(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute a single workflow through appropriate orchestrator"""
        orchestrator_type = self._get_orchestrator_for_workflow(workflow)
        
        if orchestrator_type not in self.orchestrators:
            # Create orchestrator if needed
            self.orchestrators[orchestrator_type] = await self._create_orchestrator(orchestrator_type)
        
        orchestrator = self.orchestrators[orchestrator_type]
        
        # Send workflow to orchestrator
        result = await self.send_message(
            recipient_id=orchestrator.agent_id,
            message_type='execute_workflow',
            payload={'workflow': workflow.__dict__},
            wait_for_reply=True,
            timeout=workflow.timeout
        )
        
        return result or {'error': 'Workflow execution timeout'}
    
    def _get_orchestrator_for_workflow(self, workflow: WorkflowDefinition) -> str:
        """Get orchestrator type for workflow"""
        strategy = workflow.metadata.get('strategy', '')
        return self._get_orchestrator_for_strategy(strategy)
    
    async def _create_orchestrator(self, orchestrator_type: str) -> AgentOrchestrator:
        """Create a new orchestrator instance"""
        config = {
            'type': orchestrator_type,
            'specializations': self.orchestrator_specializations.get(orchestrator_type, [])
        }
        
        orchestrator = AgentOrchestrator(
            agent_id=f"orchestrator_{orchestrator_type}",
            config=config
        )
        
        await orchestrator.initialize()
        return orchestrator
    
    async def _aggregate_results(self,
                               results: Dict[str, Any],
                               performance_targets: Dict[str, float]) -> Dict[str, Any]:
        """Aggregate results from multiple workflows"""
        aggregated = {
            'signals': [],
            'recommendations': [],
            'risk_metrics': {},
            'performance_projections': {}
        }
        
        # Collect all signals
        all_signals = []
        for workflow_id, result in results.items():
            if 'results' in result and isinstance(result['results'], dict):
                for task_result in result['results'].values():
                    if isinstance(task_result, dict) and 'signals' in task_result:
                        all_signals.extend(task_result['signals'])
        
        # Deduplicate and score signals
        unique_signals = self._deduplicate_signals(all_signals)
        scored_signals = self._score_signals(unique_signals, performance_targets)
        
        # Filter by performance targets
        filtered_signals = [
            s for s in scored_signals
            if s.get('expected_return', 0) >= performance_targets.get('min_return', 0.0)
            and s.get('risk', 1.0) <= performance_targets.get('max_risk', 1.0)
        ]
        
        aggregated['signals'] = filtered_signals
        
        # Aggregate recommendations
        all_recommendations = []
        for workflow_id, result in results.items():
            if 'recommendations' in result:
                all_recommendations.extend(result['recommendations'])
        
        aggregated['recommendations'] = self._consolidate_recommendations(all_recommendations)
        
        # Calculate aggregate risk metrics
        aggregated['risk_metrics'] = self._calculate_aggregate_risk(results)
        
        # Project performance
        aggregated['performance_projections'] = self._project_performance(
            filtered_signals,
            performance_targets
        )
        
        return aggregated
    
    def _deduplicate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate signals"""
        seen = set()
        unique = []
        
        for signal in signals:
            # Create unique key
            key = (
                signal.get('symbol', ''),
                signal.get('action', ''),
                signal.get('timeframe', '')
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(signal)
            else:
                # Merge with existing signal
                for existing in unique:
                    if (existing.get('symbol') == signal.get('symbol') and
                        existing.get('action') == signal.get('action')):
                        # Average confidence scores
                        existing['confidence'] = (
                            existing.get('confidence', 0) + signal.get('confidence', 0)
                        ) / 2
                        break
        
        return unique
    
    def _score_signals(self, 
                      signals: List[Dict[str, Any]],
                      performance_targets: Dict[str, float]) -> List[Dict[str, Any]]:
        """Score signals based on multiple factors"""
        for signal in signals:
            score = 0.0
            
            # Confidence score
            confidence = signal.get('confidence', 0.5)
            score += confidence * 0.3
            
            # Expected return vs target
            expected_return = signal.get('expected_return', 0)
            target_return = performance_targets.get('target_return', 0.02)
            if expected_return > 0:
                score += min(expected_return / target_return, 1.0) * 0.3
            
            # Risk score (inverse)
            risk = signal.get('risk', 1.0)
            max_risk = performance_targets.get('max_risk', 0.05)
            if risk < max_risk:
                score += (1 - risk / max_risk) * 0.2
            
            # Strategy diversity bonus
            strategy = signal.get('strategy', 'unknown')
            if strategy in self.strategy_performance:
                perf = self.strategy_performance[strategy]
                score += perf.get('sharpe_ratio', 0) * 0.2
            
            signal['score'] = score
        
        # Sort by score
        signals.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return signals
    
    def _consolidate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate recommendations from multiple sources"""
        # Group by symbol
        by_symbol = defaultdict(list)
        for rec in recommendations:
            symbol = rec.get('symbol', 'UNKNOWN')
            by_symbol[symbol].append(rec)
        
        consolidated = []
        for symbol, recs in by_symbol.items():
            # Aggregate recommendations
            actions = [r.get('action', 'hold') for r in recs]
            confidences = [r.get('confidence', 0.5) for r in recs]
            
            # Determine consensus action
            action_counts = defaultdict(int)
            for action in actions:
                action_counts[action] += 1
            
            consensus_action = max(action_counts.items(), key=lambda x: x[1])[0]
            avg_confidence = np.mean(confidences)
            
            consolidated.append({
                'symbol': symbol,
                'action': consensus_action,
                'confidence': avg_confidence,
                'sources': len(recs),
                'agreement': action_counts[consensus_action] / len(recs)
            })
        
        return consolidated
    
    def _calculate_aggregate_risk(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate risk metrics"""
        risk_metrics = {
            'portfolio_var': 0.0,
            'max_drawdown': 0.0,
            'correlation_risk': 0.0,
            'concentration_risk': 0.0
        }
        
        # Extract risk data from results
        all_risks = []
        for workflow_id, result in results.items():
            if 'risk_metrics' in result:
                all_risks.append(result['risk_metrics'])
        
        if all_risks:
            # Average VaR
            vars = [r.get('var', 0) for r in all_risks if 'var' in r]
            if vars:
                risk_metrics['portfolio_var'] = np.mean(vars)
            
            # Max drawdown
            drawdowns = [r.get('max_drawdown', 0) for r in all_risks if 'max_drawdown' in r]
            if drawdowns:
                risk_metrics['max_drawdown'] = max(drawdowns)
        
        return risk_metrics
    
    def _project_performance(self,
                           signals: List[Dict[str, Any]],
                           performance_targets: Dict[str, float]) -> Dict[str, float]:
        """Project performance based on signals"""
        projections = {
            'expected_return': 0.0,
            'expected_sharpe': 0.0,
            'win_probability': 0.0,
            'risk_adjusted_return': 0.0
        }
        
        if not signals:
            return projections
        
        # Calculate expected returns
        returns = [s.get('expected_return', 0) for s in signals]
        risks = [s.get('risk', 1) for s in signals]
        confidences = [s.get('confidence', 0.5) for s in signals]
        
        # Weighted average return
        weights = np.array(confidences)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        projections['expected_return'] = np.average(returns, weights=weights)
        avg_risk = np.average(risks, weights=weights)
        
        # Expected Sharpe ratio
        if avg_risk > 0:
            projections['expected_sharpe'] = projections['expected_return'] / avg_risk
        
        # Win probability based on confidence
        projections['win_probability'] = np.mean(confidences)
        
        # Risk-adjusted return
        projections['risk_adjusted_return'] = (
            projections['expected_return'] * projections['win_probability']
        )
        
        return projections
    
    async def _update_strategy_performance(self,
                                         strategies: List[str],
                                         results: Dict[str, Any]):
        """Update performance tracking for strategies"""
        perf_metrics = results.get('performance_projections', {})
        
        for strategy in strategies:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Update with exponential moving average
            current = self.strategy_performance[strategy]
            
            if 'expected_sharpe' in perf_metrics:
                current['sharpe_ratio'] = (
                    current['sharpe_ratio'] * (1 - self.adaptation_rate) +
                    perf_metrics['expected_sharpe'] * self.adaptation_rate
                )
            
            if 'win_probability' in perf_metrics:
                current['win_rate'] = (
                    current['win_rate'] * (1 - self.adaptation_rate) +
                    perf_metrics['win_probability'] * self.adaptation_rate
                )
            
            if 'expected_return' in perf_metrics:
                current['avg_return'] = (
                    current['avg_return'] * (1 - self.adaptation_rate) +
                    perf_metrics['expected_return'] * self.adaptation_rate
                )
    
    async def handle_update_market_regime(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle market regime update"""
        market_data = message.payload.get('market_data', {})
        regime = await self._detect_market_regime(market_data)
        
        return {
            'regime': regime.__dict__,
            'regime_history': [r.__dict__ for r in self.regime_history[-10:]]
        }
    
    async def handle_strategy_performance_update(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle strategy performance update"""
        strategy = message.payload.get('strategy')
        performance = message.payload.get('performance', {})
        
        if strategy and performance:
            self.strategy_performance[strategy].update(performance)
        
        return {'status': 'updated', 'strategy': strategy}
    
    async def handle_coordinate_strategies(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle strategy coordination request"""
        strategies = message.payload.get('strategies', [])
        coordination_type = message.payload.get('coordination_type', 'parallel')
        
        # Create coordination plan
        plan = {
            'strategies': strategies,
            'coordination_type': coordination_type,
            'execution_order': [],
            'dependencies': {},
            'resource_allocation': {}
        }
        
        if coordination_type == 'sequential':
            plan['execution_order'] = strategies
        elif coordination_type == 'parallel':
            plan['execution_order'] = [strategies]  # All at once
        elif coordination_type == 'hierarchical':
            # Create dependency graph
            plan['dependencies'] = self._create_strategy_dependencies(strategies)
        
        return plan
    
    async def handle_emergency_risk_override(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle emergency risk override"""
        risk_event = message.payload.get('risk_event')
        severity = message.payload.get('severity', 'medium')
        
        self.logger.warning(f"Emergency risk override: {risk_event} (severity: {severity})")
        
        # Cancel active workflows if severe
        if severity == 'critical':
            for workflow_id in list(self.active_meta_workflows.keys()):
                self.logger.info(f"Cancelling workflow {workflow_id} due to risk event")
                # Send cancellation messages
                await self.send_message(
                    recipient_id='all_orchestrators',
                    message_type='cancel_workflow',
                    payload={'workflow_id': workflow_id, 'reason': risk_event},
                    priority=MessagePriority.CRITICAL
                )
        
        return {
            'status': 'risk_override_applied',
            'cancelled_workflows': list(self.active_meta_workflows.keys()),
            'risk_event': risk_event
        }
    
    def _create_strategy_dependencies(self, strategies: List[str]) -> Dict[str, List[str]]:
        """Create dependency graph for strategies"""
        dependencies = {}
        
        # Define strategy dependencies
        strategy_deps = {
            'momentum': [],
            'mean_reversion': [],
            'sentiment_momentum': ['sentiment_analysis'],
            'options_flow': ['market_analysis'],
            'risk_arbitrage': ['fundamental_analysis', 'market_analysis']
        }
        
        for strategy in strategies:
            deps = strategy_deps.get(strategy, [])
            dependencies[strategy] = [d for d in deps if d in strategies]
        
        return dependencies
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from results"""
        metrics = {
            'expected_sharpe': 0.0,
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'signal_quality': 0.0,
            'diversification_score': 0.0
        }
        
        if 'signals' in results and results['signals']:
            returns = [s.get('expected_return', 0) for s in results['signals']]
            risks = [s.get('risk', 1) for s in results['signals']]
            
            metrics['expected_return'] = np.mean(returns) if returns else 0
            metrics['expected_volatility'] = np.mean(risks) if risks else 1
            
            if metrics['expected_volatility'] > 0:
                metrics['expected_sharpe'] = metrics['expected_return'] / metrics['expected_volatility']
            
            # Signal quality based on confidence and agreement
            confidences = [s.get('confidence', 0) for s in results['signals']]
            metrics['signal_quality'] = np.mean(confidences) if confidences else 0
            
            # Diversification score
            symbols = set(s.get('symbol') for s in results['signals'] if 'symbol' in s)
            strategies = set(s.get('strategy') for s in results['signals'] if 'strategy' in s)
            metrics['diversification_score'] = (len(symbols) * len(strategies)) / max(len(results['signals']), 1)
        
        return metrics
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request"""
        request_type = request.get('type', 'unknown')
        
        if request_type == 'execute_meta_workflow':
            return await self.handle_execute_meta_workflow(AgentMessage(
                sender_id='api',
                recipient_id=self.agent_id,
                message_type='execute_meta_workflow',
                payload=request,
                priority=MessagePriority.HIGH
            ))
        elif request_type == 'get_strategy_performance':
            return {
                'strategy_performance': dict(self.strategy_performance),
                'current_weights': self.strategy_weights,
                'current_regime': self.current_regime.__dict__ if self.current_regime else None
            }
        else:
            return await super().process_request(request) 