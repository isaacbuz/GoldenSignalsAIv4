"""
Backtesting Reporting Module - Handles report generation and visualization
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.domain.backtesting.advanced_backtest_engine import (
    BacktestMetrics,
    BacktestSignal,
    BacktestTrade,
)

logger = logging.getLogger(__name__)


class BacktestReporter:
    """Generates comprehensive backtesting reports"""

    def __init__(self, output_dir: str = "backtest_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(
        self,
        metrics: BacktestMetrics,
        trades: List[BacktestTrade],
        equity_curve: pd.Series,
        config: Dict[str, Any],
        agent_performance: Optional[Dict[str, Any]] = None,
        monte_carlo_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""

        report = {
            "metadata": self._generate_metadata(config),
            "summary": self._generate_summary(metrics),
            "detailed_metrics": self._generate_detailed_metrics(metrics),
            "trade_analysis": self._analyze_trades(trades),
            "risk_analysis": self._analyze_risk(metrics, equity_curve),
            "performance_attribution": self._attribute_performance(trades, agent_performance),
            "monte_carlo_analysis": monte_carlo_results or {},
            "recommendations": self._generate_recommendations(metrics, trades),
        }

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"backtest_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate additional outputs
        self._export_trades_csv(trades, timestamp)
        self._export_equity_curve(equity_curve, timestamp)

        logger.info(f"Backtest report saved to {report_path}")

        return report

    def _generate_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            "generated_at": datetime.now().isoformat(),
            "backtest_config": {
                "start_date": config.get("start_date"),
                "end_date": config.get("end_date"),
                "initial_capital": config.get("initial_capital"),
                "symbols": config.get("symbols"),
                "timeframe": config.get("timeframe"),
                "strategy": config.get("strategy_name", "Multi-Agent Strategy"),
            },
        }

    def _generate_summary(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Generate executive summary"""
        return {
            "total_return": f"{metrics.total_return:.2%}",
            "annualized_return": f"{metrics.annualized_return:.2%}",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{metrics.max_drawdown:.2%}",
            "win_rate": f"{metrics.win_rate:.2%}",
            "profit_factor": f"{metrics.profit_factor:.2f}",
            "total_trades": metrics.total_trades,
            "avg_trade_duration": str(metrics.avg_trade_duration),
        }

    def _generate_detailed_metrics(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Generate detailed metrics section"""
        metrics_dict = asdict(metrics)

        # Format percentages and ratios
        percentage_fields = [
            "total_return",
            "annualized_return",
            "benchmark_return",
            "alpha",
            "volatility",
            "downside_volatility",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "win_rate",
            "exposure_time",
        ]

        for field in percentage_fields:
            if field in metrics_dict:
                metrics_dict[field] = f"{metrics_dict[field]:.2%}"

        # Format ratios
        ratio_fields = [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "information_ratio",
            "profit_factor",
            "payoff_ratio",
            "beta",
        ]

        for field in ratio_fields:
            if field in metrics_dict:
                value = metrics_dict[field]
                if value == float("inf"):
                    metrics_dict[field] = "∞"
                else:
                    metrics_dict[field] = f"{value:.2f}"

        # Format currency
        currency_fields = ["avg_win", "avg_loss", "expectancy"]
        for field in currency_fields:
            if field in metrics_dict:
                metrics_dict[field] = f"${metrics_dict[field]:,.2f}"

        return metrics_dict

    def _analyze_trades(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Analyze trade patterns"""
        if not trades:
            return {"message": "No trades executed"}

        # Group by various dimensions
        by_symbol = self._group_trades_by_symbol(trades)
        by_hour = self._group_trades_by_hour(trades)
        by_day_of_week = self._group_trades_by_day_of_week(trades)
        by_duration = self._group_trades_by_duration(trades)

        # Best and worst trades
        sorted_trades = sorted(trades, key=lambda t: t.pnl_percent, reverse=True)
        best_trades = [self._trade_summary(t) for t in sorted_trades[:5]]
        worst_trades = [self._trade_summary(t) for t in sorted_trades[-5:]]

        return {
            "by_symbol": by_symbol,
            "by_hour": by_hour,
            "by_day_of_week": by_day_of_week,
            "by_duration": by_duration,
            "best_trades": best_trades,
            "worst_trades": worst_trades,
            "consecutive_wins": self._max_consecutive_wins(trades),
            "consecutive_losses": self._max_consecutive_losses(trades),
        }

    def _analyze_risk(self, metrics: BacktestMetrics, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze risk characteristics"""
        returns = equity_curve.pct_change().dropna()

        # Distribution analysis
        return {
            "return_distribution": {
                "mean": f"{returns.mean():.4f}",
                "std": f"{returns.std():.4f}",
                "skew": f"{returns.skew():.2f}",
                "kurtosis": f"{returns.kurtosis():.2f}",
            },
            "risk_metrics": {
                "var_95": f"{metrics.var_95:.2%}",
                "cvar_95": f"{metrics.cvar_95:.2%}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "max_drawdown_duration": f"{metrics.max_drawdown_duration} days",
            },
            "tail_risk": self._analyze_tail_risk(returns),
            "correlation_to_market": self._calculate_market_correlation(returns),
        }

    def _attribute_performance(
        self, trades: List[BacktestTrade], agent_performance: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Attribute performance to different factors"""
        if not agent_performance:
            return {"message": "No agent performance data available"}

        # Calculate contribution by agent
        agent_contributions = {}

        for trade in trades:
            primary_agent = self._get_primary_agent(trade.signal.agent_scores)
            if primary_agent not in agent_contributions:
                agent_contributions[primary_agent] = {
                    "trades": 0,
                    "total_pnl": 0,
                    "win_rate": 0,
                    "wins": 0,
                }

            agent_contributions[primary_agent]["trades"] += 1
            agent_contributions[primary_agent]["total_pnl"] += trade.pnl
            if trade.pnl > 0:
                agent_contributions[primary_agent]["wins"] += 1

        # Calculate win rates
        for agent, stats in agent_contributions.items():
            if stats["trades"] > 0:
                stats["win_rate"] = stats["wins"] / stats["trades"]
                stats["avg_pnl"] = stats["total_pnl"] / stats["trades"]

        return {"agent_contributions": agent_contributions, "agent_performance": agent_performance}

    def _generate_recommendations(
        self, metrics: BacktestMetrics, trades: List[BacktestTrade]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []

        # Performance-based recommendations
        if metrics.sharpe_ratio < 1:
            recommendations.append(
                {
                    "type": "risk_adjusted_returns",
                    "priority": "high",
                    "recommendation": "Sharpe ratio is below 1. Consider improving risk-adjusted returns by reducing position sizes or improving signal quality.",
                }
            )

        if metrics.win_rate < 0.4:
            recommendations.append(
                {
                    "type": "win_rate",
                    "priority": "high",
                    "recommendation": "Win rate is below 40%. Review entry criteria and consider more selective signal generation.",
                }
            )

        if metrics.max_drawdown > 0.2:
            recommendations.append(
                {
                    "type": "risk_management",
                    "priority": "critical",
                    "recommendation": f"Maximum drawdown of {metrics.max_drawdown:.1%} exceeds 20%. Implement stricter risk controls.",
                }
            )

        # Trade analysis recommendations
        if trades:
            avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0])
            avg_win = np.mean([t.pnl for t in trades if t.pnl > 0])

            if abs(avg_loss) > avg_win * 2:
                recommendations.append(
                    {
                        "type": "position_sizing",
                        "priority": "high",
                        "recommendation": "Average loss is more than 2x average win. Consider implementing better stop-loss strategies.",
                    }
                )

        return recommendations

    def _group_trades_by_symbol(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Group trades by symbol"""
        symbol_stats = {}

        for trade in trades:
            symbol = trade.signal.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"count": 0, "total_pnl": 0, "wins": 0, "total_return": 0}

            symbol_stats[symbol]["count"] += 1
            symbol_stats[symbol]["total_pnl"] += trade.pnl
            symbol_stats[symbol]["total_return"] += trade.pnl_percent
            if trade.pnl > 0:
                symbol_stats[symbol]["wins"] += 1

        # Calculate averages
        for symbol, stats in symbol_stats.items():
            stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
            stats["avg_return"] = (
                stats["total_return"] / stats["count"] if stats["count"] > 0 else 0
            )

        return symbol_stats

    def _group_trades_by_hour(self, trades: List[BacktestTrade]) -> Dict[int, int]:
        """Group trades by hour of day"""
        hour_distribution = {}

        for trade in trades:
            hour = trade.entry_time.hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1

        return dict(sorted(hour_distribution.items()))

    def _group_trades_by_day_of_week(self, trades: List[BacktestTrade]) -> Dict[str, int]:
        """Group trades by day of week"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_distribution = {day: 0 for day in days}

        for trade in trades:
            day_name = days[trade.entry_time.weekday()]
            day_distribution[day_name] += 1

        return day_distribution

    def _group_trades_by_duration(self, trades: List[BacktestTrade]) -> Dict[str, int]:
        """Group trades by duration buckets"""
        duration_buckets = {
            "< 1 hour": 0,
            "1-4 hours": 0,
            "4-24 hours": 0,
            "1-3 days": 0,
            "3-7 days": 0,
            "> 7 days": 0,
        }

        for trade in trades:
            hours = trade.time_in_trade.total_seconds() / 3600

            if hours < 1:
                duration_buckets["< 1 hour"] += 1
            elif hours < 4:
                duration_buckets["1-4 hours"] += 1
            elif hours < 24:
                duration_buckets["4-24 hours"] += 1
            elif hours < 72:
                duration_buckets["1-3 days"] += 1
            elif hours < 168:
                duration_buckets["3-7 days"] += 1
            else:
                duration_buckets["> 7 days"] += 1

        return duration_buckets

    def _trade_summary(self, trade: BacktestTrade) -> Dict[str, Any]:
        """Create trade summary"""
        return {
            "symbol": trade.signal.symbol,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "pnl": f"${trade.pnl:,.2f}",
            "pnl_percent": f"{trade.pnl_percent:.2%}",
            "duration": str(trade.time_in_trade),
            "exit_reason": trade.exit_reason,
        }

    def _max_consecutive_wins(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum consecutive wins"""
        max_wins = 0
        current_wins = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0

        return max_wins

    def _max_consecutive_losses(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum consecutive losses"""
        max_losses = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl <= 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def _analyze_tail_risk(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze tail risk"""
        sorted_returns = returns.sort_values()
        n = len(returns)

        return {
            "worst_1pct": float(sorted_returns.iloc[: int(n * 0.01)].mean()),
            "worst_5pct": float(sorted_returns.iloc[: int(n * 0.05)].mean()),
            "best_1pct": float(sorted_returns.iloc[-int(n * 0.01) :].mean()),
            "best_5pct": float(sorted_returns.iloc[-int(n * 0.05) :].mean()),
        }

    def _calculate_market_correlation(self, returns: pd.Series) -> float:
        """Calculate correlation to market (placeholder)"""
        # In real implementation, would correlate with SPY or market index
        return 0.0

    def _get_primary_agent(self, agent_scores: Dict[str, float]) -> str:
        """Get primary agent from scores"""
        if not agent_scores:
            return "unknown"
        return max(agent_scores.items(), key=lambda x: x[1])[0]

    def _export_trades_csv(self, trades: List[BacktestTrade], timestamp: str):
        """Export trades to CSV"""
        if not trades:
            return

        trades_data = []
        for trade in trades:
            trades_data.append(
                {
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "symbol": trade.signal.symbol,
                    "action": trade.signal.action,
                    "entry_price": trade.signal.entry_price,
                    "exit_price": trade.exit_price,
                    "position_size": trade.signal.position_size,
                    "pnl": trade.pnl,
                    "pnl_percent": trade.pnl_percent,
                    "exit_reason": trade.exit_reason,
                    "confidence": trade.signal.confidence,
                }
            )

        df = pd.DataFrame(trades_data)
        csv_path = self.output_dir / f"trades_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Trades exported to {csv_path}")

    def _export_equity_curve(self, equity_curve: pd.Series, timestamp: str):
        """Export equity curve to CSV"""
        csv_path = self.output_dir / f"equity_curve_{timestamp}.csv"
        equity_curve.to_csv(csv_path, header=["equity"])

        logger.info(f"Equity curve exported to {csv_path}")
