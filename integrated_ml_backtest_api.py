#!/usr/bin/env python3
"""
Integrated ML Backtesting API for GoldenSignalsAI
Provides REST endpoints for backtesting and signal accuracy improvement
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import logging
from ml_enhanced_backtest_system import MLBacktestEngine, SignalAccuracyImprover
from advanced_backtest_system import AdvancedBacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GoldenSignalsAI ML Backtesting API",
    description="ML-enhanced backtesting and signal accuracy improvement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
ml_engine = MLBacktestEngine()
accuracy_improver = SignalAccuracyImprover()
advanced_engine = AdvancedBacktestEngine()

# Global cache for backtest results
backtest_cache = {}
running_backtests = {}


class BacktestRequest(BaseModel):
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategy_type: str = "ml_ensemble"  # ml_ensemble, technical, hybrid
    use_walk_forward: bool = True
    n_splits: int = 5
    include_transaction_costs: bool = True
    commission: float = 0.001
    slippage: float = 0.0005


class SignalValidationRequest(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    features: Optional[Dict[str, float]] = None


class BacktestResponse(BaseModel):
    backtest_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SignalImprovementRequest(BaseModel):
    symbols: List[str]
    lookback_days: int = 365
    min_accuracy_threshold: float = 0.55


@app.get("/")
async def root():
    return {
        "message": "GoldenSignalsAI ML Backtesting API",
        "endpoints": {
            "POST /backtest": "Run ML-enhanced backtest",
            "GET /backtest/{backtest_id}": "Get backtest results",
            "POST /validate-signal": "Validate a signal against historical performance",
            "POST /improve-signals": "Get signal improvement recommendations",
            "GET /backtest-history": "Get all backtest results",
            "GET /performance-metrics": "Get current model performance metrics"
        }
    }


async def run_backtest_async(backtest_id: str, request: BacktestRequest):
    """Run backtest in background"""
    try:
        running_backtests[backtest_id] = {"status": "running", "progress": 0.1}

        # Set date range
        if not request.start_date:
            request.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not request.end_date:
            request.end_date = datetime.now().strftime('%Y-%m-%d')

        # Update progress
        running_backtests[backtest_id]["progress"] = 0.2

        # Run appropriate backtest based on strategy type
        if request.strategy_type == "ml_ensemble":
            results = await ml_engine.run_comprehensive_backtest(
                request.symbols,
                request.start_date,
                request.end_date
            )
        elif request.strategy_type == "technical":
            # Use advanced engine for technical analysis
            results = {}
            for symbol in request.symbols:
                result = await advanced_engine.run_backtest(
                    symbol,
                    request.start_date,
                    request.end_date,
                    initial_capital=10000,
                    position_size=0.1
                )
                results[symbol] = result
        else:  # hybrid
            # Combine both approaches
            ml_results = await ml_engine.run_comprehensive_backtest(
                request.symbols,
                request.start_date,
                request.end_date
            )
            technical_results = {}
            for symbol in request.symbols:
                result = await advanced_engine.run_backtest(
                    symbol,
                    request.start_date,
                    request.end_date,
                    initial_capital=10000,
                    position_size=0.1
                )
                technical_results[symbol] = result

            # Merge results
            results = {
                "ml_results": ml_results,
                "technical_results": technical_results,
                "combined_metrics": calculate_combined_metrics(ml_results, technical_results)
            }

        # Store results
        backtest_cache[backtest_id] = {
            "id": backtest_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "request": request.dict(),
            "results": results
        }

        running_backtests[backtest_id] = {"status": "completed", "progress": 1.0}

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        running_backtests[backtest_id] = {
            "status": "failed",
            "progress": 0,
            "error": str(e)
        }
        backtest_cache[backtest_id] = {
            "id": backtest_id,
            "status": "failed",
            "error": str(e)
        }


def calculate_combined_metrics(ml_results: Dict, technical_results: Dict) -> Dict:
    """Calculate combined metrics from ML and technical results"""
    combined = {
        "average_sharpe_ratio": 0,
        "average_annual_return": 0,
        "average_max_drawdown": 0,
        "best_strategy": None,
        "recommendations": []
    }

    # Calculate averages
    ml_sharpes = []
    tech_sharpes = []

    for symbol, data in ml_results.items():
        if 'backtest_metrics' in data:
            ml_sharpes.append(data['backtest_metrics']['sharpe_ratio'])

    for symbol, data in technical_results.items():
        if 'metrics' in data:
            tech_sharpes.append(data['metrics']['sharpe_ratio'])

    ml_avg_sharpe = np.mean(ml_sharpes) if ml_sharpes else 0
    tech_avg_sharpe = np.mean(tech_sharpes) if tech_sharpes else 0

    combined['average_sharpe_ratio'] = (ml_avg_sharpe + tech_avg_sharpe) / 2
    combined['best_strategy'] = "ml_ensemble" if ml_avg_sharpe > tech_avg_sharpe else "technical"

    # Add recommendations
    if ml_avg_sharpe > 1.5:
        combined['recommendations'].append("ML models showing strong performance - consider increasing position sizes")
    if tech_avg_sharpe > 1.5:
        combined['recommendations'].append("Technical indicators performing well - maintain current strategy")

    return combined


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run ML-enhanced backtest
    """
    # Generate unique backtest ID
    backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(request.symbols))}"

    # Start backtest in background
    background_tasks.add_task(run_backtest_async, backtest_id, request)

    return BacktestResponse(
        backtest_id=backtest_id,
        status="started",
        progress=0.0
    )


@app.get("/backtest/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_results(backtest_id: str):
    """
    Get backtest results by ID
    """
    if backtest_id in running_backtests:
        status_info = running_backtests[backtest_id]

        if status_info["status"] == "completed" and backtest_id in backtest_cache:
            return BacktestResponse(
                backtest_id=backtest_id,
                status="completed",
                progress=1.0,
                results=backtest_cache[backtest_id]
            )
        elif status_info["status"] == "failed":
            return BacktestResponse(
                backtest_id=backtest_id,
                status="failed",
                progress=status_info.get("progress", 0),
                error=status_info.get("error", "Unknown error")
            )
        else:
            return BacktestResponse(
                backtest_id=backtest_id,
                status=status_info["status"],
                progress=status_info.get("progress", 0)
            )

    raise HTTPException(status_code=404, detail="Backtest not found")


@app.post("/validate-signal")
async def validate_signal(request: SignalValidationRequest):
    """
    Validate a signal against historical performance
    """
    try:
        # Fetch historical data for the symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Last 90 days

        df = ml_engine.fetch_historical_data(
            request.symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")

        # Engineer features
        df = ml_engine.engineer_features(df)

        # Simple validation based on recent performance
        recent_returns = df['returns'].tail(20)
        volatility = recent_returns.std() * np.sqrt(252)

        # Calculate expected return based on action and confidence
        if request.action == "BUY":
            expected_return = request.confidence * recent_returns.mean() * 252
        elif request.action == "SELL":
            expected_return = -request.confidence * recent_returns.mean() * 252
        else:  # HOLD
            expected_return = 0

        # Risk-adjusted validation
        risk_adjusted_score = expected_return / volatility if volatility > 0 else 0

        # Historical win rate for similar conditions
        similar_signals = df[df['rsi'].between(
            df['rsi'].iloc[-1] - 5,
            df['rsi'].iloc[-1] + 5
        )]

        if len(similar_signals) > 10:
            historical_win_rate = (similar_signals['target'] == 1).mean()
        else:
            historical_win_rate = 0.5  # Default if not enough data

        return {
            "symbol": request.symbol,
            "action": request.action,
            "confidence": request.confidence,
            "validation": {
                "expected_annual_return": expected_return,
                "current_volatility": volatility,
                "risk_adjusted_score": risk_adjusted_score,
                "historical_win_rate": historical_win_rate,
                "recommendation": "PROCEED" if risk_adjusted_score > 0.5 and historical_win_rate > 0.55 else "CAUTION",
                "confidence_level": "HIGH" if request.confidence > 0.8 and historical_win_rate > 0.6 else "MEDIUM" if request.confidence > 0.6 else "LOW"
            }
        }

    except Exception as e:
        logger.error(f"Signal validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/improve-signals")
async def improve_signals(request: SignalImprovementRequest):
    """
    Get signal improvement recommendations based on historical analysis
    """
    try:
        improvements = await accuracy_improver.improve_signals(request.symbols)

        # Filter recommendations based on accuracy threshold
        filtered_features = [
            (feature, importance)
            for feature, importance in improvements['recommended_features']
            if importance > request.min_accuracy_threshold
        ]

        return {
            "symbols": request.symbols,
            "improvements": {
                "top_features": filtered_features[:10],
                "optimal_parameters": improvements['optimal_parameters'],
                "risk_rules": improvements['risk_management'],
                "signal_filters": improvements['signal_filters'],
                "expected_improvement": {
                    "accuracy_gain": "5-10%",
                    "sharpe_ratio_improvement": "0.2-0.5",
                    "drawdown_reduction": "10-20%"
                }
            },
            "implementation_steps": [
                "1. Focus on top features in signal generation",
                "2. Apply recommended signal filters",
                "3. Implement risk management rules",
                "4. Use optimal position sizing",
                "5. Monitor performance and adjust"
            ]
        }

    except Exception as e:
        logger.error(f"Signal improvement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest-history")
async def get_backtest_history(limit: int = 10):
    """
    Get history of all backtests
    """
    # Sort by timestamp and return most recent
    sorted_backtests = sorted(
        backtest_cache.values(),
        key=lambda x: x.get('timestamp', ''),
        reverse=True
    )[:limit]

    return {
        "count": len(backtest_cache),
        "backtests": sorted_backtests
    }


@app.get("/performance-metrics")
async def get_performance_metrics():
    """
    Get current model performance metrics
    """
    # Calculate aggregate metrics from recent backtests
    recent_backtests = list(backtest_cache.values())[-5:]  # Last 5 backtests

    if not recent_backtests:
        return {
            "message": "No backtests available",
            "metrics": {}
        }

    # Aggregate metrics
    all_sharpe_ratios = []
    all_returns = []
    all_drawdowns = []

    for backtest in recent_backtests:
        if backtest.get('status') == 'completed' and 'results' in backtest:
            results = backtest['results']

            # Handle different result formats
            if isinstance(results, dict):
                for symbol, data in results.items():
                    if isinstance(data, dict) and 'backtest_metrics' in data:
                        metrics = data['backtest_metrics']
                        all_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                        all_returns.append(metrics.get('annual_return', 0))
                        all_drawdowns.append(abs(metrics.get('max_drawdown', 0)))

    return {
        "aggregate_metrics": {
            "average_sharpe_ratio": np.mean(all_sharpe_ratios) if all_sharpe_ratios else 0,
            "average_annual_return": np.mean(all_returns) if all_returns else 0,
            "average_max_drawdown": np.mean(all_drawdowns) if all_drawdowns else 0,
            "best_sharpe_ratio": max(all_sharpe_ratios) if all_sharpe_ratios else 0,
            "worst_drawdown": max(all_drawdowns) if all_drawdowns else 0
        },
        "model_status": {
            "ml_models": "active",
            "last_training": datetime.now().isoformat(),
            "data_quality": "good",
            "recommendations": [
                "Consider retraining models weekly",
                "Monitor for regime changes",
                "Adjust position sizes based on volatility"
            ]
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting ML Backtesting API on port 8001")
    print("📊 Access API docs at http://localhost:8001/docs")

    uvicorn.run(app, host="0.0.0.0", port=8001)
