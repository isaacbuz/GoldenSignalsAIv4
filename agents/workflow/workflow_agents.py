"""
Workflow orchestration utilities for managing trading cycles and processes.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, time
import pytz
from src.monitoring.monitoring_agents import AIMonitor
from src.strategy.strategy_utils import StrategyTuner

logger = logging.getLogger(__name__)

class TradingWorkflow:
    """Base class for trading workflows."""
    
    def __init__(self, name: str):
        """Initialize the workflow.
        
        Args:
            name (str): Workflow name.
        """
        self.name = name
        self.monitor = AIMonitor()
        self.strategy_tuner = StrategyTuner()
        self.is_running = False
        
    async def start(self):
        """Start the workflow."""
        self.is_running = True
        logger.info({"message": f"Started workflow: {self.name}"})
        
    async def stop(self):
        """Stop the workflow."""
        self.is_running = False
        logger.info({"message": f"Stopped workflow: {self.name}"})
        
    async def execute(self):
        """Execute one cycle of the workflow."""
        raise NotImplementedError

class DailyTradingCycle(TradingWorkflow):
    """Manages the daily trading cycle."""
    
    def __init__(
        self,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        timezone: str = "America/New_York"
    ):
        """Initialize the daily trading cycle.
        
        Args:
            market_open (time): Market opening time.
            market_close (time): Market closing time.
            timezone (str): Market timezone.
        """
        super().__init__(name="DailyTradingCycle")
        self.market_open = market_open
        self.market_close = market_close
        self.timezone = pytz.timezone(timezone)
        self.pre_market_tasks = []
        self.market_hours_tasks = []
        self.post_market_tasks = []
        
    def add_pre_market_task(self, task: callable):
        """Add a pre-market task.
        
        Args:
            task (callable): Task to execute before market open.
        """
        self.pre_market_tasks.append(task)
        
    def add_market_hours_task(self, task: callable):
        """Add a market hours task.
        
        Args:
            task (callable): Task to execute during market hours.
        """
        self.market_hours_tasks.append(task)
        
    def add_post_market_task(self, task: callable):
        """Add a post-market task.
        
        Args:
            task (callable): Task to execute after market close.
        """
        self.post_market_tasks.append(task)
        
    async def execute(self):
        """Execute one cycle of the daily trading workflow."""
        try:
            current_time = datetime.now(self.timezone).time()
            
            # Pre-market tasks
            if current_time < self.market_open:
                logger.info({"message": "Executing pre-market tasks"})
                for task in self.pre_market_tasks:
                    try:
                        await task()
                    except Exception as e:
                        logger.error({"message": f"Pre-market task failed: {str(e)}"})
                        
            # Market hours tasks
            elif self.market_open <= current_time <= self.market_close:
                logger.info({"message": "Executing market hours tasks"})
                for task in self.market_hours_tasks:
                    try:
                        await task()
                    except Exception as e:
                        logger.error({"message": f"Market hours task failed: {str(e)}"})
                        
            # Post-market tasks
            else:
                logger.info({"message": "Executing post-market tasks"})
                for task in self.post_market_tasks:
                    try:
                        await task()
                    except Exception as e:
                        logger.error({"message": f"Post-market task failed: {str(e)}"})
                        
            # Update system metrics
            await self.monitor.update_metrics()
            
        except Exception as e:
            logger.error({"message": f"Daily trading cycle failed: {str(e)}"})

# Create a global instance for easy access
daily_trading_cycle = DailyTradingCycle() 