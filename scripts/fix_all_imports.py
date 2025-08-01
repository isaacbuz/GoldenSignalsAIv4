#!/usr/bin/env python3
"""
Fix all import issues across the codebase systematically.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix common import issues in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Define import replacements
    replacements = [
        # Fix agents.services imports
        (r'from agents\.services\.', 'from src.services.'),
        (r'import agents\.services\.', 'import src.services.'),

        # Fix agents.core.dependencies imports
        (r'from agents\.core\.dependencies', 'from src.core.dependencies'),

        # Fix agents.core.agent_factory imports
        (r'from agents\.core\.agent_factory', 'from agents.agent_factory'),

        # Fix relative imports in test files
        (r'from src\.main import app', '''import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app'''),

        # Fix non-existent modules
        (r'from agents\.predictive import', '# # from agents.predictive import'),
        (r'from src\.agents\.predictive import', '# # from src.agents.predictive import'),
        (r'from agents\.legacy_backend_agents', '# # from agents.legacy_backend_agents'),

        # Fix sentiment imports
        (r'from agents\.sentiment import (\w+)', r'from agents.core.sentiment.\1 import \1'),
        (r'from agents\.news import (\w+)', r'from agents.core.sentiment.news_agent import \1'),

        # Fix technical imports
        (r'from agents\.technical import', 'from agents.core.technical import'),
        (r'from agents\.rsi import RSIAgent', 'from agents.core.technical.momentum.rsi_agent import RSIAgent'),

        # Fix base imports
        (r'from src\.base import', 'from agents.base_agent import'),

        # Fix API endpoint imports
        (r'from src\.api\.endpoints import', 'from src.api.v1 import'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Special handling for test files that import from src.main
    if 'tests/' in str(file_path) and 'import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app' in content and 'sys.path.append' not in content:
        # Add the import fix at the beginning of the file
        lines = content.split('\n')
        import_index = -1
        for i, line in enumerate(lines):
            if 'import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app' in line:
                import_index = i
                break

        if import_index >= 0:
            lines[import_index] = '''import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import app'''
            content = '\n'.join(lines)

    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix imports across the entire codebase."""
    print("Fixing imports across the codebase...")

    # Directories to process
    directories = [
        'src',
        'agents',
        'tests',
        'scripts',
        'mcp_servers'
    ]

    fixed_files = []

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for root, dirs, files in os.walk(directory):
            # Skip virtual environments and cache
            if any(skip in root for skip in ['.venv', '__pycache__', '.git', 'htmlcov', 'node_modules']):
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        if fix_imports_in_file(file_path):
                            fixed_files.append(file_path)
                            print(f"✅ Fixed: {file_path}")
                    except Exception as e:
                        print(f"❌ Error fixing {file_path}: {e}")

    print(f"\n✅ Fixed {len(fixed_files)} files")

    if fixed_files:
        print("\nFixed files:")
        for f in sorted(fixed_files)[:20]:  # Show first 20
            print(f"  - {f}")
        if len(fixed_files) > 20:
            print(f"  ... and {len(fixed_files) - 20} more")

if __name__ == "__main__":
    main()
