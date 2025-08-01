#!/usr/bin/env python3
"""Fix all import and module errors systematically."""

import os
import re
import ast
from pathlib import Path
from typing import List, Set, Tuple

class ImportFixer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.missing_modules = set()
        self.fixed_files = []
        self.created_files = []

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        return list(self.root_dir.rglob("*.py"))

    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return imports

    def check_module_exists(self, module_path: str) -> bool:
        """Check if a module exists in the project."""
        # Convert module path to file path
        parts = module_path.split('.')

        # Check for package
        package_path = self.root_dir / Path(*parts)
        if package_path.exists() and package_path.is_dir():
            init_file = package_path / "__init__.py"
            return init_file.exists()

        # Check for module file
        module_file = self.root_dir / Path(*parts[:-1]) / f"{parts[-1]}.py"
        return module_file.exists()

    def create_missing_init_files(self):
        """Create missing __init__.py files."""
        directories_to_check = [
            "src/infrastructure",
            "src/infrastructure/database",
            "src/infrastructure/monitoring",
            "src/infrastructure/integration",
            "src/infrastructure/cache",
            "src/api/rag",
            "src/api/v2",
            "tests/mocks",
            "tests/fixtures",
            "tests/unit/agents",
            "tests/integration/services",
            "agents/core",
            "agents/meta",
            "agents/optimization",
            "agents/research/ml",
            "agents/research/ml/patterns",
            "agents/research/ml/regime",
            "agents/research/ml/options"
        ]

        for dir_path in directories_to_check:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    self.created_files.append(str(init_file))
                    print(f"Created: {init_file}")

    def fix_common_import_patterns(self, file_path: Path):
        """Fix common import pattern issues."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            original_content = content

            # Common import fixes
            patterns = [
                # Fix relative imports in tests
                (r'from \.\.\.(\w+)', r'from agents.\1'),
                (r'from \.\.(\w+)', r'from src.\1'),

                # Fix base imports
                (r'from agents.base import', r'from agents.base import'),
                (r'from (\w+)\.base import', r'from agents.base import'),

                # Fix infrastructure imports
                (r'from infrastructure\.', r'from src.infrastructure.'),
                (r'from services\.', r'from src.services.'),
                (r'from api\.', r'from src.api.'),

                # Fix test imports
                (r'from tests\.fixtures import', r'from tests.fixtures.conftest import'),
                (r'from mocks\.', r'from tests.mocks.'),

                # Fix ML model imports
                (r'from ml\.models\.', r'from src.ml.models.'),
                (r'from ml_training\.', r'from ml_training.'),

                # Fix websocket imports
                (r'from websocket\.', r'from src.websocket.'),

                # Fix domain imports
                (r'from domain\.', r'from src.domain.'),
                (r'from trading\.', r'from src.domain.trading.'),
            ]

            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

            # Fix specific module references
            specific_fixes = {
                'from src.infrastructure.database.enhanced_query_optimizer import': 'from src.infrastructure.database.enhanced_query_optimizer import',
                'from src.websocket.scalable_manager import': 'from src.websocket.scalable_manager import',
                'from src.services.database_optimization_service import': 'from src.services.database_optimization_service import',
                'from src.ml.models.market_data import MarketData': 'from src.ml.models.market_data import MarketData',
                'from src.ml.models.signals import Signal': 'from src.ml.models.signals import Signal',
            }

            for old, new in specific_fixes.items():
                content = content.replace(old, new)

            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                print(f"Fixed imports in: {file_path}")

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

    def create_mock_modules(self):
        """Create mock modules for testing."""
        mock_modules = {
            "tests/mocks/redis_mock.py": '''"""Mock Redis client for testing."""

class MockRedisClient:
    def __init__(self):
        self.data = {}
        self.pubsub_messages = []
        self.pubsub_subscribers = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    async def hget(self, name, key):
        hash_data = self.data.get(name, {})
        return hash_data.get(key)

    async def hset(self, name, key, value):
        if name not in self.data:
            self.data[name] = {}
        self.data[name][key] = value
        return 1

    async def publish(self, channel, message):
        self.pubsub_messages.append((channel, message))
        return len(self.pubsub_subscribers.get(channel, []))

    def pubsub(self):
        return MockPubSub(self)

    async def close(self):
        pass

class MockPubSub:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.channels = set()

    async def subscribe(self, *channels):
        self.channels.update(channels)

    async def unsubscribe(self, *channels):
        for channel in channels:
            self.channels.discard(channel)

    async def get_message(self, timeout=None):
        return None
''',
            "tests/mocks/database_mock.py": '''"""Mock database manager for testing."""

from typing import List, Optional, Dict, Any
from datetime import datetime

class MockDatabaseManager:
    def __init__(self):
        self.signals = []
        self.market_data = []
        self.portfolios = {}
        self.users = {}

    async def store_signal(self, signal):
        self.signals.append(signal)
        return signal.signal_id if hasattr(signal, 'signal_id') else len(self.signals)

    async def get_signals(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        signals = self.signals
        if symbol:
            signals = [s for s in signals if s.get('symbol') == symbol]
        return signals[:limit]

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        return self.market_data[:limit]

    async def update_agent_performance(self, agent_id: str, metrics: Dict[str, Any]):
        pass

    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]):
        pass

    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return None
''',
            "tests/mocks/__init__.py": '''"""Mock objects for testing."""

from .redis_mock import MockRedisClient, MockPubSub
from .database_mock import MockDatabaseManager

__all__ = ['MockRedisClient', 'MockPubSub', 'MockDatabaseManager']
'''
        }

        for file_path, content in mock_modules.items():
            full_path = self.root_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if not full_path.exists():
                with open(full_path, 'w') as f:
                    f.write(content)
                self.created_files.append(str(full_path))
                print(f"Created mock: {full_path}")

    def run(self):
        """Run the import fixer."""
        print("Starting import fix process...")
        print("=" * 50)

        # Step 1: Create missing __init__.py files
        print("\n1. Creating missing __init__.py files...")
        self.create_missing_init_files()

        # Step 2: Create mock modules
        print("\n2. Creating mock modules...")
        self.create_mock_modules()

        # Step 3: Fix imports in all Python files
        print("\n3. Fixing imports in Python files...")
        python_files = self.find_python_files()

        for file_path in python_files:
            # Skip __pycache__ and .git directories
            if '__pycache__' in str(file_path) or '.git' in str(file_path):
                continue

            self.fix_common_import_patterns(file_path)

        # Step 4: Generate report
        print("\n" + "=" * 50)
        print("Import Fix Summary:")
        print(f"- Created {len(self.created_files)} new files")
        print(f"- Fixed imports in {len(self.fixed_files)} files")

        if self.created_files:
            print("\nCreated files:")
            for f in self.created_files[:10]:  # Show first 10
                print(f"  - {f}")
            if len(self.created_files) > 10:
                print(f"  ... and {len(self.created_files) - 10} more")

        if self.fixed_files:
            print("\nFixed files:")
            for f in self.fixed_files[:10]:  # Show first 10
                print(f"  - {f}")
            if len(self.fixed_files) > 10:
                print(f"  ... and {len(self.fixed_files) - 10} more")

def main():
    """Main execution."""
    root_dir = Path(__file__).parent.parent
    fixer = ImportFixer(root_dir)
    fixer.run()

if __name__ == "__main__":
    main()
