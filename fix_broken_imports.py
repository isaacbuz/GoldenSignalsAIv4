#!/usr/bin/env python3
"""
Fix broken imports after archive removal
Part of Issue #209 - Codebase Consolidation
"""

import os
import re

# Files with broken imports and their fixes
IMPORT_FIXES = {
    'tests/test_blender_agent.py': [
        ('from archive.', 'from agents.'),
        ('import archive.', 'import agents.')
    ],
    'tests/test_rsi_agent.py': [
        ('from archive.', 'from agents.'),
        ('import archive.', 'import agents.')
    ],
    'agents/adapters/legacy/volume/obv.py': [
        ('from archive.', 'from agents.'),
        ('import archive.', 'import agents.')
    ],
    'src/automation/trade_executor.py': [
        ('from archive.', 'from src.'),
        ('import archive.', 'import src.')
    ],
    'src/automation/retrain_scheduler.py': [
        ('from archive.', 'from src.'),
        ('import archive.', 'import src.')
    ]
}

def fix_imports():
    """Fix broken imports in files"""
    print("🔧 Fixing broken imports after archive removal")
    print("="*50)
    
    fixed_count = 0
    
    for filename, replacements in IMPORT_FIXES.items():
        if os.path.exists(filename):
            print(f"\n📝 Fixing {filename}...")
            
            with open(filename, 'r') as f:
                content = f.read()
            
            original_content = content
            for old_pattern, new_pattern in replacements:
                content = content.replace(old_pattern, new_pattern)
            
            if content != original_content:
                with open(filename, 'w') as f:
                    f.write(content)
                print(f"  ✅ Fixed imports")
                fixed_count += 1
            else:
                print(f"  ℹ️ No archive imports found")
        else:
            print(f"\n⚠️ File not found: {filename}")
    
    print("\n" + "="*50)
    print(f"✅ Fixed {fixed_count} files")
    print("\n✅ All broken imports resolved!")

if __name__ == "__main__":
    fix_imports() 