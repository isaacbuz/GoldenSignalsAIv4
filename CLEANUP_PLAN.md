# GoldenSignalsAI Cleanup Plan

## Overview
This document outlines the cleanup strategy for redundant and unused code in the GoldenSignalsAI project.

## Files to Remove

### 1. Test Files in Root Directory (22 files)
These should be moved to the `tests/` directory or removed if redundant:
- `test_after_hours_demo.py`
- `test_after_hours.py`
- `test_all_agents.py`
- `test_backend_format.py`
- `test_hybrid_simple.py`
- `test_hybrid_system.py`
- `test_live_data_and_backtest.py`
- `test_live_data_simple.py`
- `test_live_data.py`
- `test_local_db.py`
- `test_phase2_agents.py`
- `test_phase3_agents.py`
- `test_quick_setup.py`
- `test_rate_limits.py`
- `test_signal_generation.py`
- `test_simple_server.py`
- `test_system_health.py`
- `test_system.py`
- `test_yfinance_connectivity.py`

### 2. Demo Files in Root Directory (7 files)
These were used for testing and can be removed:
- `demo_integrated_system.py`
- `demo_live_backtest_fixed.py`
- `demo_live_backtest.py`
- `demo_live_data_professional.py`
- `demo_precise_signals.py`
- `demo_rate_limit_solution.py`
- `demo_signal_system.py`

### 3. Redundant Backend Files
- `simple_backend_backup.py` - Backup of simple_backend.py
- `src/main_simple_v2.py` - Old version

### 4. Duplicate Backtesting Engines
Multiple implementations exist:
- `backtesting/backtest_engine.py`
- `backtesting/enhanced_backtest_engine.py`
- `backtesting/simple_backtest.py`
- `agents/research/backtesting/backtest_engine.py`
- `src/domain/backtesting/backtest_engine.py`

### 5. Simple/Test Agents
These were for testing and can be consolidated:
- `agents/core/options/simple_options_flow_agent.py`
- `agents/core/sentiment/simple_sentiment_agent.py`
- `agents/core/technical/simple_working_agent.py`
- `agents/meta/simple_consensus_agent.py`

### 6. Archive Directory
The `archive/` directory contains legacy code that should be reviewed

## Files to Integrate

### 1. Backtesting System
Consolidate into one comprehensive backtesting engine:
- Keep: `src/domain/backtesting/backtest_engine.py` (most integrated)
- Merge useful features from other implementations

### 2. ML Training
Consolidate ML training scripts:
- `ml_training/train_demo_models.py` - Keep if useful
- `ml_training/test_data_*.py` files - Remove test files

### 3. MCP Servers
Keep all MCP servers as they're actively used

## Recommended Actions

1. **Create tests/ subdirectories**:
   - `tests/integration/`
   - `tests/unit/`
   - `tests/performance/`
   - `tests/agents/`

2. **Consolidate configuration**:
   - Remove duplicate config files
   - Use environment variables consistently

3. **Clean up imports**:
   - Remove unused imports
   - Standardize import paths

4. **Documentation**:
   - Update README files to reflect current structure
   - Remove outdated documentation

## Execution Order

1. Move test files to proper directories
2. Remove demo files
3. Remove backup files
4. Consolidate backtesting engines
5. Clean up simple/test agents
6. Review and clean archive directory
7. Update imports and documentation

## Files to Keep

- `simple_backend.py` - Currently in use
- `start.sh` - Master startup script
- All MCP servers
- Production-ready agents
- Current configuration files 