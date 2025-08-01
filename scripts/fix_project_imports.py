#!/usr/bin/env python3
"""Fix imports only in project files, not in virtual environments."""

import os
import re
from pathlib import Path
from typing import List, Set

class ProjectImportFixer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.fixed_files = []
        self.created_files = []

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        path_str = str(file_path)

        # Skip virtual environments and cache
        skip_dirs = ['.venv', 'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']
        for skip_dir in skip_dirs:
            if skip_dir in path_str:
                return False

        # Only process Python files in our project
        valid_dirs = ['agents', 'src', 'tests', 'scripts', 'ml_training']
        for valid_dir in valid_dirs:
            if path_str.startswith(str(self.root_dir / valid_dir)):
                return True

        return False

    def create_missing_init_files(self):
        """Create missing __init__.py files in project directories."""
        directories_to_check = [
            "src/infrastructure",
            "src/infrastructure/monitoring",
            "src/infrastructure/integration",
            "tests/mocks",
            "tests/fixtures",
            "tests/unit/agents",
            "tests/integration/services"
        ]

        for dir_path in directories_to_check:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    self.created_files.append(str(init_file))
                    print(f"Created: {init_file}")

    def fix_test_imports(self, file_path: Path):
        """Fix imports specifically in test files."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            original_content = content

            # Test-specific fixes
            test_fixes = [
                # Fix infrastructure imports in tests
                (r'from src\.infrastructure\.database\.enhanced_query_optimizer import',
                 'from src.infrastructure.database.enhanced_query_optimizer import'),
                (r'from src\.websocket\.scalable_manager import',
                 'from src.websocket.scalable_manager import'),

                # Fix mock imports
                (r'from tests\.mocks import', 'from tests.mocks import'),
                (r'from tests\.fixtures import', 'from tests.fixtures.conftest import'),

                # Fix service imports
                (r'from src\.services\.database_optimization_service import',
                 'from src.services.database_optimization_service import'),
            ]

            for pattern, replacement in test_fixes:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                print(f"Fixed test imports in: {file_path.name}")

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

    def run(self):
        """Run the targeted import fixer."""
        print("Starting targeted import fix...")
        print("=" * 50)

        # Step 1: Create missing __init__.py files
        print("\n1. Creating missing __init__.py files...")
        self.create_missing_init_files()

        # Step 2: Fix imports in test files
        print("\n2. Fixing imports in test files...")
        test_files = list((self.root_dir / "tests").rglob("test_*.py"))

        for test_file in test_files:
            if self.should_process_file(test_file):
                self.fix_test_imports(test_file)

        # Step 3: Summary
        print("\n" + "=" * 50)
        print("Import Fix Summary:")
        print(f"- Created {len(self.created_files)} new files")
        print(f"- Fixed imports in {len(self.fixed_files)} files")

        if self.created_files:
            print("\nCreated files:")
            for f in self.created_files:
                print(f"  - {f}")

def main():
    """Main execution."""
    root_dir = Path(__file__).parent.parent
    fixer = ProjectImportFixer(root_dir)
    fixer.run()

if __name__ == "__main__":
    main()
