#!/bin/bash

# Core domain migrations
mv agents/technical/* agents/core/technical/ 2>/dev/null
mv agents/fundamental/* agents/core/fundamental/ 2>/dev/null
mv agents/sentiment/* agents/core/sentiment/ 2>/dev/null
mv agents/risk/* agents/core/risk/ 2>/dev/null
mv agents/timing/* agents/core/timing/ 2>/dev/null
mv agents/portfolio/* agents/core/portfolio/ 2>/dev/null

# Infrastructure migrations
mv agents/data_sources/* agents/infrastructure/data_sources/ 2>/dev/null
mv agents/integration/* agents/infrastructure/integration/ 2>/dev/null
mv agents/monitoring/* agents/infrastructure/monitoring/ 2>/dev/null
mv agents/integration_utils.py agents/infrastructure/integration/
mv agents/monitoring_agents.py agents/infrastructure/monitoring/

# Research migrations
mv agents/backtesting/* agents/research/backtesting/ 2>/dev/null
mv agents/ml/* agents/research/ml/ 2>/dev/null
mv agents/predictive/* agents/research/ml/ 2>/dev/null
mv agents/research_agents.py agents/research/
mv agents/backtesting_utils.py agents/research/backtesting/
mv agents/options_backtesting.py agents/research/backtesting/

# Common migrations
mv agents/base/* agents/common/base/ 2>/dev/null
mv agents/base_agent.py agents/common/base/
mv agents/agent_registry.py agents/common/registry/
mv agents/factory.py agents/common/registry/
mv agents/strategy_utils.py agents/common/utils/
mv agents/workflow_agents.py agents/common/utils/

# Experimental migrations
mv agents/grok/* agents/experimental/grok/ 2>/dev/null
mv agents/vision/* agents/experimental/vision/ 2>/dev/null
mv agents/adaptive/* agents/experimental/adaptive/ 2>/dev/null
mv agents/grok_agents.py agents/experimental/grok/

# Clean up empty directories
find agents -type d -empty -delete

echo "Migration completed!" 