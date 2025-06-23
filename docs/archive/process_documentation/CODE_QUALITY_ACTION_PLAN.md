# Code Quality Action Plan - GoldenSignalsAI V2

## Overview
This document provides concrete actions to improve Code Organization, Type Safety, and Test Coverage.

## 1. Code Organization (High Priority) 🔴

### Current Issues
- Multiple signal generators with overlapping functionality
- Inconsistent module structure
- Unclear service boundaries
- Mixed responsibilities in classes

### Action Items

#### Week 1: Service Layer Consolidation
1. **Consolidate Signal Generators**
   ```
   Current:
   - signal_generator_simple.py
   - integrated_signal_generator.py  
   - precise_signal_demo.py
   
   Target:
   - src/services/signal_service.py (unified implementation)
   ```

2. **Create Clear Service Boundaries**
   ```
   src/
   ├── api/           # API endpoints only
   ├── services/      # Business logic
   ├── repositories/  # Data access
   ├── models/        # Domain models
   └── utils/         # Shared utilities
   ```

3. **Implement Dependency Injection**
   ```python
   # src/core/container.py
   from dependency_injector import containers, providers
   
   class Container(containers.DeclarativeContainer):
       config = providers.Configuration()
       
       # Data layer
       market_data_repo = providers.Singleton(
           MarketDataRepository,
           api_key=config.api_key
       )
       
       # Service layer
       signal_service = providers.Singleton(
           SignalService,
           market_data_repo=market_data_repo
       )
   ```

### Implementation Script

```python
# refactor_code_organization.py
#!/usr/bin/env python3
"""
Script to reorganize code structure
"""

import os
import shutil
from pathlib import Path

def reorganize_services():
    """Reorganize service layer"""
    # Create new structure
    new_dirs = [
        "src/repositories",
        "src/services/signals",
        "src/services/market",
        "src/services/portfolio",
        "src/core/di"
    ]
    
    for dir_path in new_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Move files to appropriate locations
    moves = {
        "signal_generator_simple.py": "src/services/signals/legacy_simple.py",
        "integrated_signal_generator.py": "src/services/signals/legacy_integrated.py",
        "src/services/signal_generation_engine.py": "src/services/signals/signal_service.py"
    }
    
    for src, dst in moves.items():
        if Path(src).exists():
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")

if __name__ == "__main__":
    reorganize_services()
```

## 2. Type Safety (Medium Priority) ⚠️

### Current Issues
- No type hints in most functions
- Dict[str, Any] used extensively
- No runtime type validation
- Missing protocol definitions

### Action Items

#### Week 1: Core Type Definitions
1. **Create Type Definitions**
   ```python
   # src/types/market.py
   from typing import TypedDict, Literal, Protocol
   from datetime import datetime
   from decimal import Decimal
   
   class MarketData(TypedDict):
       symbol: str
       price: Decimal
       volume: int
       timestamp: datetime
       
   SignalAction = Literal["BUY", "SELL", "HOLD"]
   
   class TradingSignal(TypedDict):
       id: str
       symbol: str
       action: SignalAction
       confidence: float
       price: Decimal
       timestamp: datetime
   ```

2. **Add Type Hints to All Functions**
   ```python
   # Before
   def generate_signal(symbol, data):
       # ...
       
   # After
   def generate_signal(
       symbol: str, 
       data: pd.DataFrame
   ) -> Optional[TradingSignal]:
       # ...
   ```

3. **Use Pydantic for Runtime Validation**
   ```python
   # src/models/signals.py
   from pydantic import BaseModel, validator
   from decimal import Decimal
   from datetime import datetime
   
   class SignalRequest(BaseModel):
       symbol: str
       timeframe: str = "1d"
       
       @validator('symbol')
       def symbol_must_be_uppercase(cls, v):
           return v.upper()
   
   class SignalResponse(BaseModel):
       signal: TradingSignal
       metadata: Dict[str, Any]
       generated_at: datetime
   ```

### Type Safety Script

```python
# add_type_hints.py
#!/usr/bin/env python3
"""
Script to add type hints to Python files
"""

import ast
import os
from pathlib import Path
from typing import List, Set

def find_untyped_functions(file_path: Path) -> List[str]:
    """Find functions without type hints"""
    untyped = []
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function has return type
            if node.returns is None:
                untyped.append(f"{file_path}:{node.lineno} - {node.name}")
    
    return untyped

def scan_project() -> None:
    """Scan project for untyped functions"""
    untyped_functions = []
    
    for py_file in Path("src").rglob("*.py"):
        untyped = find_untyped_functions(py_file)
        untyped_functions.extend(untyped)
    
    print(f"Found {len(untyped_functions)} untyped functions")
    for func in untyped_functions[:10]:  # Show first 10
        print(f"  - {func}")

if __name__ == "__main__":
    scan_project()
```

## 3. Test Coverage (Medium Priority) ⚠️

### Current Issues
- Limited unit tests
- No integration tests
- Missing test fixtures
- No coverage reporting

### Action Items

#### Week 1: Test Infrastructure
1. **Set Up pytest Configuration**
   ```ini
   # pytest.ini
   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = 
       --cov=src
       --cov-report=html
       --cov-report=term-missing
       --cov-fail-under=80
   ```

2. **Create Test Fixtures**
   ```python
   # tests/conftest.py
   import pytest
   from datetime import datetime
   import pandas as pd
   
   @pytest.fixture
   def sample_market_data():
       """Sample market data for testing"""
       return pd.DataFrame({
           'open': [100, 101, 102],
           'high': [105, 106, 107],
           'low': [99, 100, 101],
           'close': [104, 105, 106],
           'volume': [1000, 1100, 1200]
       })
   
   @pytest.fixture
   def mock_signal_service(mocker):
       """Mocked signal service"""
       service = mocker.Mock()
       service.generate_signal.return_value = {
           'symbol': 'AAPL',
           'action': 'BUY',
           'confidence': 0.85
       }
       return service
   ```

3. **Write Unit Tests for Critical Functions**
   ```python
   # tests/unit/test_signal_service.py
   import pytest
   from src.services.signals import SignalService
   
   class TestSignalService:
       def test_generate_signal_buy_condition(self, sample_market_data):
           service = SignalService()
           signal = service.generate_signal('AAPL', sample_market_data)
           
           assert signal is not None
           assert signal['action'] in ['BUY', 'SELL', 'HOLD']
           assert 0 <= signal['confidence'] <= 1
       
       def test_invalid_symbol_raises_error(self):
           service = SignalService()
           with pytest.raises(ValueError):
               service.generate_signal('', pd.DataFrame())
   ```

### Test Coverage Script

```python
# setup_tests.py
#!/usr/bin/env python3
"""
Script to set up comprehensive testing
"""

import os
from pathlib import Path

def create_test_structure():
    """Create test directory structure"""
    test_dirs = [
        "tests/unit/api",
        "tests/unit/services",
        "tests/unit/repositories",
        "tests/integration",
        "tests/e2e",
        "tests/fixtures"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        init_file = Path(dir_path) / "__init__.py"
        init_file.touch()
    
    # Create conftest.py
    conftest_content = '''import pytest
import pandas as pd
from datetime import datetime, timezone

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'api_key': 'test_key',
        'cache_ttl': 60,
        'max_retries': 3
    }
'''
    
    with open("tests/conftest.py", "w") as f:
        f.write(conftest_content)
    
    print("✅ Test structure created")

def generate_sample_tests():
    """Generate sample test files"""
    # Unit test example
    unit_test = '''import pytest
from src.services.signal_generation_engine import SignalGenerationEngine

class TestSignalGenerationEngine:
    def test_initialization(self, test_config):
        engine = SignalGenerationEngine()
        assert engine.cache_ttl == 300
        assert engine.ml_model is not None
    
    def test_generate_signal_with_valid_data(self, sample_market_data):
        engine = SignalGenerationEngine()
        signal = engine._analyze_market_data('AAPL', sample_market_data)
        
        assert signal is not None
        assert 'action' in signal
        assert 'confidence' in signal
    
    def test_cache_functionality(self):
        engine = SignalGenerationEngine()
        # First call
        signal1 = engine.generate_signal('AAPL')
        # Second call should use cache
        signal2 = engine.generate_signal('AAPL')
        
        assert signal1 == signal2
'''
    
    with open("tests/unit/services/test_signal_generation.py", "w") as f:
        f.write(unit_test)
    
    print("✅ Sample tests generated")

if __name__ == "__main__":
    create_test_structure()
    generate_sample_tests()
    print("\n📊 Run 'pytest --cov' to see coverage report")
```

## Quick Start Commands

```bash
# 1. Code Organization
python refactor_code_organization.py

# 2. Type Safety
pip install mypy pydantic
python add_type_hints.py
mypy src/ --install-types

# 3. Test Coverage
pip install pytest pytest-cov pytest-mock
python setup_tests.py
pytest --cov=src --cov-report=html

# Run all quality checks
make quality-check
```

## Makefile Addition

```makefile
# Add to Makefile
quality-check:
	@echo "🔍 Running quality checks..."
	@echo "1. Type checking..."
	mypy src/ --install-types --non-interactive
	@echo "2. Running tests..."
	pytest --cov=src --cov-fail-under=60
	@echo "3. Code formatting..."
	black src/ tests/
	@echo "✅ Quality checks complete"

fix-types:
	python add_type_hints.py
	mypy src/ --install-types

test-watch:
	ptw -- --cov=src --cov-report=term-missing
```

## Expected Outcomes

### After Implementation
1. **Code Organization**
   - Single source of truth for each service
   - Clear separation of concerns
   - Easier to navigate and maintain
   - 50% reduction in duplicate code

2. **Type Safety**
   - 100% type coverage for public APIs
   - Runtime validation for external inputs
   - Catch bugs at development time
   - Better IDE support and autocomplete

3. **Test Coverage**
   - 80%+ test coverage
   - Automated test runs on commit
   - Confidence in refactoring
   - Faster bug detection

## Next Steps
1. Start with Code Organization (Week 1)
2. Add Type Safety incrementally (Week 2)
3. Build tests for new code first (Week 3)
4. Backfill tests for existing code (Week 4) 