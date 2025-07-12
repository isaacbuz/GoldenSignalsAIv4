#!/usr/bin/env python3
"""
Fix missing 'import os' in test files that use os.path.
"""

import os
import re

def fix_file(file_path):
    """Fix missing import os in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file uses os but doesn't import it
    if 'os.path' in content or 'os.dirname' in content:
        lines = content.split('\n')
        
        # Check if os is already imported
        has_import_os = False
        for line in lines:
            if re.match(r'^import os\s*$', line) or 'import os' in line:
                has_import_os = True
                break
        
        if not has_import_os:
            # Find where to insert import os
            insert_index = 0
            
            # Look for existing imports
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_index = i
                    # If we find import sys, put os right after it
                    if 'import sys' in line:
                        insert_index = i + 1
                        break
            
            # Insert import os
            lines.insert(insert_index, 'import os')
            
            # Write back
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
            return True
    
    return False

def main():
    """Fix all test files with missing import os."""
    fixed_files = []
    
    # Walk through test directory
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    if fix_file(file_path):
                        fixed_files.append(file_path)
                        print(f"✅ Fixed: {file_path}")
                except Exception as e:
                    print(f"❌ Error fixing {file_path}: {e}")
    
    print(f"\n✅ Fixed {len(fixed_files)} files")

if __name__ == "__main__":
    main() 