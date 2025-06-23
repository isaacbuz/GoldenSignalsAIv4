# Priority GitHub Issues for GoldenSignalsAI Refactoring

Copy and paste these into GitHub Issues manually if preferred.

---

## Issue 1: 🔧 Code Organization - Implement Service Layer Refactoring

**Labels:** `refactoring`, `priority:high`, `enhancement`

## Overview
Consolidate duplicate code and implement clean service architecture.

## Tasks
- [ ] Consolidate 5 signal generators into 1 unified service
- [ ] Implement dependency injection using the created container
- [ ] Move remaining services to new structure (signals/, market/, portfolio/, risk/)
- [ ] Update all imports to use new paths

## Impact
- 50% reduction in duplicate code
- Clear separation of concerns
- Easier maintenance

Related: CODE_QUALITY_ACTION_PLAN.md

---

## Issue 2: 📊 Type Safety - Add Type Hints to Critical Functions

**Labels:** `type-safety`, `priority:high`, `enhancement`

## Overview
Improve type safety across the codebase. Current coverage: 60.3% (669/1109 functions).

## Priority Functions to Type
1. `src/main_v2.py:330 - main()`
2. `src/working_server.py:216 - main()`
3. `src/services/signal_service.py` - All public methods
4. `src/services/market/data_service.py` - All public methods

## Goals
- Reach 80% type coverage by end of week
- 100% coverage for public APIs
- Add runtime validation with Pydantic

Related: add_type_hints.py analysis

---

## Issue 3: 🧪 Test Coverage - Write Tests for Signal Generation

**Labels:** `testing`, `priority:high`, `quality`

## Overview
Implement comprehensive tests for critical signal generation logic.

## Priority Tests
- [ ] Unit tests for SignalGenerationEngine
- [ ] Integration tests for signal filtering pipeline
- [ ] Test timezone handling (regression test for fixed bug)
- [ ] Test cache functionality
- [ ] Test ML model integration

## Infrastructure Ready
- pytest configured with async support
- Test fixtures created
- Sample tests provided

Goal: 80% test coverage

Related: tests/unit/services/test_signal_generation.py

---

## Issue 4: 🐛 Verify Timezone Fix Across All Services

**Labels:** `bug`, `priority:high`, `verification`

## Overview
We fixed a critical timezone bug, but need to ensure it's handled correctly everywhere.

## Completed
- ✅ Fixed in signal_service.py
- ✅ Fixed in data_quality_validator.py
- ✅ Fixed in market/quality_validator.py

## TODO
- [ ] Audit all datetime operations in the codebase
- [ ] Ensure all use timezone-aware datetime
- [ ] Add timezone handling to coding standards
- [ ] Create utility functions for common datetime operations

Related: TIMEZONE_ISSUES_REPORT.md (426 issues found)

---

## Issue 5: 📚 Update Import Statements Project-Wide

**Labels:** `refactoring`, `priority:medium`, `task`

## Overview
After code reorganization, many imports need updating.

## Changes Needed
- `signal_generation_engine` → `signals.signal_service`
- `signal_filtering_pipeline` → `signals.signal_filter`
- `data_quality_validator` → `market.quality_validator`

## Files to Update
- All files importing from old service locations
- Test files
- Documentation examples

Use grep to find: `grep -r "from src.services.signal_generation_engine"`

---

## Issue 6: 🏗️ Implement Dependency Injection Framework

**Labels:** `architecture`, `priority:medium`, `enhancement`

## Overview
We created a DI container, now implement it across the application.

## Tasks
- [ ] Wire up container in main application
- [ ] Replace manual instantiation with DI
- [ ] Add configuration management
- [ ] Create factories for complex objects
- [ ] Document DI patterns

## Benefits
- Easier testing (mock injection)
- Better configuration management
- Cleaner initialization

Related: src/core/di/container.py

---

## Issue 7: 📈 Performance Optimization Using New Architecture

**Labels:** `performance`, `priority:medium`, `enhancement`

## Overview
Leverage the new clean architecture for performance improvements.

## Opportunities
- [ ] Implement proper caching strategy at repository level
- [ ] Add connection pooling for data sources
- [ ] Optimize signal generation with batch processing
- [ ] Profile and optimize hot paths
- [ ] Add performance metrics

## Metrics to Track
- Signal generation time
- API response time
- Cache hit rates
- Memory usage

---

## Issue 8: 📖 API Documentation with OpenAPI/Swagger

**Labels:** `documentation`, `priority:low`, `enhancement`

## Overview
Document all APIs using OpenAPI specification.

## Tasks
- [ ] Add OpenAPI schemas to FastAPI endpoints
- [ ] Document request/response models
- [ ] Add example requests
- [ ] Generate client SDKs
- [ ] Create API versioning strategy

## Benefits
- Auto-generated documentation
- Client SDK generation
- Better API testing

---

