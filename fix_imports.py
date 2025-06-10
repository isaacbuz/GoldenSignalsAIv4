#!/usr/bin/env python3
"""
Automated Import Migration Script for GoldenSignalsAI V3
Fixes broken imports after backend consolidation
"""

import os
import re
import subprocess
from pathlib import Path

def fix_imports():
    """Fix all broken import statements"""
    
    # Define import mapping
    import_mappings = {
        # Domain imports
        r'from domain\.': 'from src.domain.',
        r'import domain\.': 'import src.domain.',
        
        # Application imports  
        r'from application\.': 'from src.application.',
        r'import application\.': 'import src.application.',
        
        # Backend imports (map to appropriate src locations)
        r'from backend\.agents\.': 'from archive.legacy_backend_agents.',
        r'from backend\.api\.': 'from src.legacy_api.',
        r'from backend\.db\.': 'from src.legacy_db.',
        r'from backend\.models\.': 'from src.legacy_models.',
        r'from backend\.': 'from src.legacy_config.',
    }
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip excluded directories
        if any(excluded in root for excluded in ['venv', '.git', 'node_modules']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"🔧 Processing {len(python_files)} Python files...")
    
    fixed_count = 0
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import mappings
            for old_pattern, new_replacement in import_mappings.items():
                content = re.sub(old_pattern, new_replacement, content)
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_count += 1
                print(f"✅ Fixed imports in: {file_path}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n🎉 Import migration complete!")
    print(f"📊 Files modified: {fixed_count}")
    print(f"📋 Total files processed: {len(python_files)}")

if __name__ == "__main__":
    fix_imports() 