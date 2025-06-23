#!/bin/bash

# Create priority GitHub issues from refactoring work
# Requires: gh auth login

echo "Creating priority issues for GoldenSignalsAI refactoring..."

echo "[1/8] Creating: 🔧 Code Organization - Implement Service Layer Refactoring"
gh issue create --title "🔧 Code Organization - Implement Service Layer Refactoring" --body "## Overview\nConsolidate duplicate code and implement clean service architecture.\n\n## Tasks\n- [ ] Consolidate 5 signal generators into 1 unified service\n- [ ] Implement dependency injection using the created container\n- [ ] Move remaining services to new structure (signals/, market/, portfolio/, risk/)\n- [ ] Update all imports to use new paths\n\n## Impact\n- 50% reduction in duplicate code\n- Clear separation of concerns\n- Easier maintenance\n\nRelated: CODE_QUALITY_ACTION_PLAN.md" --label "refactoring,priority:high,enhancement"
sleep 2  # Avoid rate limiting

echo "[2/8] Creating: 📊 Type Safety - Add Type Hints to Critical Functions"
gh issue create --title "📊 Type Safety - Add Type Hints to Critical Functions" --body "## Overview\nImprove type safety across the codebase. Current coverage: 60.3% (669/1109 functions).\n\n## Priority Functions to Type\n1. `src/main_v2.py:330 - main()`\n2. `src/working_server.py:216 - main()`\n3. `src/services/signal_service.py` - All public methods\n4. `src/services/market/data_service.py` - All public methods\n\n## Goals\n- Reach 80% type coverage by end of week\n- 100% coverage for public APIs\n- Add runtime validation with Pydantic\n\nRelated: add_type_hints.py analysis" --label "type-safety,priority:high,enhancement"
sleep 2  # Avoid rate limiting

echo "[3/8] Creating: 🧪 Test Coverage - Write Tests for Signal Generation"
gh issue create --title "🧪 Test Coverage - Write Tests for Signal Generation" --body "## Overview\nImplement comprehensive tests for critical signal generation logic.\n\n## Priority Tests\n- [ ] Unit tests for SignalGenerationEngine\n- [ ] Integration tests for signal filtering pipeline\n- [ ] Test timezone handling (regression test for fixed bug)\n- [ ] Test cache functionality\n- [ ] Test ML model integration\n\n## Infrastructure Ready\n- pytest configured with async support\n- Test fixtures created\n- Sample tests provided\n\nGoal: 80% test coverage\n\nRelated: tests/unit/services/test_signal_generation.py" --label "testing,priority:high,quality"
sleep 2  # Avoid rate limiting

echo "[4/8] Creating: 🐛 Verify Timezone Fix Across All Services"
gh issue create --title "🐛 Verify Timezone Fix Across All Services" --body "## Overview\nWe fixed a critical timezone bug, but need to ensure it's handled correctly everywhere.\n\n## Completed\n- ✅ Fixed in signal_service.py\n- ✅ Fixed in data_quality_validator.py\n- ✅ Fixed in market/quality_validator.py\n\n## TODO\n- [ ] Audit all datetime operations in the codebase\n- [ ] Ensure all use timezone-aware datetime\n- [ ] Add timezone handling to coding standards\n- [ ] Create utility functions for common datetime operations\n\nRelated: TIMEZONE_ISSUES_REPORT.md (426 issues found)" --label "bug,priority:high,verification"
sleep 2  # Avoid rate limiting

echo "[5/8] Creating: 📚 Update Import Statements Project-Wide"
gh issue create --title "📚 Update Import Statements Project-Wide" --body "## Overview\nAfter code reorganization, many imports need updating.\n\n## Changes Needed\n- `signal_generation_engine` → `signals.signal_service`\n- `signal_filtering_pipeline` → `signals.signal_filter`\n- `data_quality_validator` → `market.quality_validator`\n\n## Files to Update\n- All files importing from old service locations\n- Test files\n- Documentation examples\n\nUse grep to find: `grep -r \"from src.services.signal_generation_engine\"`" --label "refactoring,priority:medium,task"
sleep 2  # Avoid rate limiting

echo "[6/8] Creating: 🏗️ Implement Dependency Injection Framework"
gh issue create --title "🏗️ Implement Dependency Injection Framework" --body "## Overview\nWe created a DI container, now implement it across the application.\n\n## Tasks\n- [ ] Wire up container in main application\n- [ ] Replace manual instantiation with DI\n- [ ] Add configuration management\n- [ ] Create factories for complex objects\n- [ ] Document DI patterns\n\n## Benefits\n- Easier testing (mock injection)\n- Better configuration management\n- Cleaner initialization\n\nRelated: src/core/di/container.py" --label "architecture,priority:medium,enhancement"
sleep 2  # Avoid rate limiting

echo "[7/8] Creating: 📈 Performance Optimization Using New Architecture"
gh issue create --title "📈 Performance Optimization Using New Architecture" --body "## Overview\nLeverage the new clean architecture for performance improvements.\n\n## Opportunities\n- [ ] Implement proper caching strategy at repository level\n- [ ] Add connection pooling for data sources\n- [ ] Optimize signal generation with batch processing\n- [ ] Profile and optimize hot paths\n- [ ] Add performance metrics\n\n## Metrics to Track\n- Signal generation time\n- API response time\n- Cache hit rates\n- Memory usage" --label "performance,priority:medium,enhancement"
sleep 2  # Avoid rate limiting

echo "[8/8] Creating: 📖 API Documentation with OpenAPI/Swagger"
gh issue create --title "📖 API Documentation with OpenAPI/Swagger" --body "## Overview\nDocument all APIs using OpenAPI specification.\n\n## Tasks\n- [ ] Add OpenAPI schemas to FastAPI endpoints\n- [ ] Document request/response models\n- [ ] Add example requests\n- [ ] Generate client SDKs\n- [ ] Create API versioning strategy\n\n## Benefits\n- Auto-generated documentation\n- Client SDK generation\n- Better API testing" --label "documentation,priority:low,enhancement"
sleep 2  # Avoid rate limiting

echo "✅ All priority issues created!"
