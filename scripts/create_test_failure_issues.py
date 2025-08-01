#!/usr/bin/env python3
"""Create GitHub issues for all test failures."""

import os
import requests
import json
from datetime import datetime

# GitHub API configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = 'isaacbuz'
REPO_NAME = 'GoldenSignalsAIv4'

headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

# Test failure issues to create
test_issues = [
    {
        "title": "🔧 Fix RSI Agent Abstract Class Implementation",
        "body": """## Problem
RSI Agent and other technical agents fail to instantiate due to missing abstract methods from BaseAgent.

## Error
```
TypeError: Can't instantiate abstract class RSIAgent with abstract methods analyze, get_required_data_types
```

## Root Cause
The BaseAgent class has abstract methods that aren't implemented in child classes:
- `analyze()`
- `get_required_data_types()`

## Solution
1. Implement the missing abstract methods in RSIAgent
2. Update all other agent classes that inherit from BaseAgent
3. Ensure proper method signatures match the base class

## Acceptance Criteria
- [ ] All agent classes can be instantiated
- [ ] RSI agent tests pass
- [ ] No abstract method errors
""",
        "labels": ["bug", "agents", "testing", "P0"]
    },
    {
        "title": "🗂️ Fix Backend Unit Test Import Errors",
        "body": """## Problem
Backend unit tests are failing due to import errors after archive cleanup.

## Affected Tests
- tests/unit/*
- Missing modules from removed archives
- Changed import paths

## Solution
1. Update all import statements in test files
2. Create missing test fixtures
3. Update conftest.py with proper paths

## Acceptance Criteria
- [ ] All backend unit tests run without import errors
- [ ] Test discovery works properly
- [ ] Fixtures are properly loaded
""",
        "labels": ["bug", "testing", "backend", "P0"]
    },
    {
        "title": "🔌 Fix Backend Integration Test Setup",
        "body": """## Problem
Integration tests fail due to database and service setup issues.

## Issues
- Database connections not properly initialized
- Redis not mocked/started for tests
- Service dependencies not injected

## Solution
1. Create proper test database setup
2. Add Redis test container or mock
3. Update integration test fixtures
4. Ensure proper teardown

## Acceptance Criteria
- [ ] Integration tests connect to test database
- [ ] Redis operations are properly tested
- [ ] All services initialize correctly
- [ ] Tests clean up after themselves
""",
        "labels": ["bug", "testing", "integration", "P0"]
    },
    {
        "title": "📊 Implement Missing Performance Tests",
        "body": """## Problem
Performance test suite is missing or incomplete.

## Required Tests
- API endpoint response time tests
- Agent processing performance
- Database query performance
- Memory usage tests
- Concurrent request handling

## Solution
1. Create performance test framework
2. Implement benchmark tests for critical paths
3. Add performance regression detection
4. Create performance baseline

## Acceptance Criteria
- [ ] Performance test suite runs
- [ ] Benchmarks are recorded
- [ ] Performance regressions are detected
- [ ] Results are reported clearly
""",
        "labels": ["testing", "performance", "P1"]
    },
    {
        "title": "🎨 Fix Frontend Test Infrastructure",
        "body": """## Problem
Frontend tests fail due to missing infrastructure.

## Issues
- test_logs directory not created
- npm test configuration issues
- Cypress E2E setup incomplete

## Solution
1. Create test_logs directory structure
2. Fix npm test scripts in package.json
3. Configure Cypress properly
4. Add frontend unit test setup

## Acceptance Criteria
- [ ] Frontend unit tests run successfully
- [ ] E2E tests execute in headless mode
- [ ] Test logs are properly generated
- [ ] Coverage reports work
""",
        "labels": ["bug", "testing", "frontend", "P0"]
    },
    {
        "title": "🤖 Fix ML Model Test Data",
        "body": """## Problem
ML training tests fail because model files and test data are missing.

## Missing Items
- Test model files
- Training data fixtures
- Model evaluation metrics
- Test predictions

## Solution
1. Create minimal test models
2. Generate synthetic test data
3. Mock ML operations where appropriate
4. Add model serialization tests

## Acceptance Criteria
- [ ] ML tests run without file errors
- [ ] Test models are properly loaded
- [ ] Training tests complete successfully
- [ ] Model predictions are tested
""",
        "labels": ["bug", "testing", "ml", "P1"]
    },
    {
        "title": "🏗️ Create Missing Test Infrastructure",
        "body": """## Problem
Various test infrastructure components are missing.

## Missing Components
- test_logs/ directory
- Test configuration files
- Mock data generators
- Test utilities

## Solution
1. Create all missing directories
2. Add test configuration templates
3. Implement test data factories
4. Create shared test utilities

## Acceptance Criteria
- [ ] All required directories exist
- [ ] Test configurations are loaded
- [ ] Mock data is generated properly
- [ ] Utilities are reusable
""",
        "labels": ["testing", "infrastructure", "P0"]
    },
    {
        "title": "📈 Increase Test Coverage to 60%",
        "body": """## Problem
Current test coverage is ~2%, target is 60%.

## Priority Modules for Coverage
1. Core agents (30+ files with 0% coverage)
2. API endpoints
3. Services layer
4. Domain logic

## Strategy
1. Start with high-value, high-risk modules
2. Write unit tests for all public methods
3. Add integration tests for workflows
4. Use coverage reports to find gaps

## Acceptance Criteria
- [ ] Overall coverage reaches 60%
- [ ] All critical paths have tests
- [ ] Coverage report is generated in CI
- [ ] No untested public APIs
""",
        "labels": ["testing", "coverage", "P1"]
    }
]

# Create issues
created_issues = []
for issue_data in test_issues:
    url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues'
    response = requests.post(url, headers=headers, json=issue_data)

    if response.status_code == 201:
        issue = response.json()
        created_issues.append({
            'number': issue['number'],
            'title': issue['title'],
            'url': issue['html_url']
        })
        print(f"✅ Created issue #{issue['number']}: {issue['title']}")
    else:
        print(f"❌ Failed to create issue: {issue_data['title']}")
        print(f"   Error: {response.status_code} - {response.text}")

# Save created issues
if created_issues:
    with open('test_failure_issues.json', 'w') as f:
        json.dump(created_issues, f, indent=2)

    print(f"\n✅ Successfully created {len(created_issues)} issues")
    print("\nCreated Issues:")
    for issue in created_issues:
        print(f"  - #{issue['number']}: {issue['title']}")
        print(f"    {issue['url']}")
else:
    print("\n❌ No issues were created")
