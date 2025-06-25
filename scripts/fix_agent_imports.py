#!/usr/bin/env python3
"""
Fix agent import issues in the codebase
"""

import os
import re
import sys

def fix_agent_imports():
    """Fix all agent import issues"""
    print("Fixing agent import issues...")
    
    # Define import replacements
    replacements = [
        # Fix UnifiedBaseAgent imports
        (r'from agents\.core\.unified_base_agent import.*', 'from agents.base import BaseAgent'),
        (r'class \w+\(UnifiedBaseAgent\):', lambda m: m.group(0).replace('UnifiedBaseAgent', 'BaseAgent')),
        
        # Fix AgentType, MessagePriority, AgentMessage imports
        (r'from agents\.base import BaseAgent$', 'from agents.base import BaseAgent, Signal, SignalSource'),
        
        # Remove undefined imports
        (r'', ''),
        
        # Fix technical agent imports
        (r'from agents\.technical\.rsi_agent import RSIAgent', 'from agents.core.technical.momentum.rsi_agent import RSIAgent'),
    ]
    
    # Walk through all Python files
    for root, dirs, files in os.walk('.'):
        # Skip virtual environments and cache directories
        if any(skip in root for skip in ['.venv', '__pycache__', '.git', 'htmlcov']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply replacements
                    for pattern, replacement in replacements:
                        if callable(replacement):
                            content = re.sub(pattern, replacement, content)
                        else:
                            content = re.sub(pattern, replacement, content)
                    
                    # Special handling for momentum_agent.py
                    if 'momentum_agent.py' in filepath and 'UnifiedBaseAgent' in content:
                        # Replace UnifiedBaseAgent with BaseAgent
                        content = content.replace('UnifiedBaseAgent', 'BaseAgent')
                        
                        # Remove undefined types
                        content = content.replace(', AgentType.TECHNICAL,', ',')
                        content = re.sub(r'super\(\).__init__\(agent_id, AgentType\.\w+, config\)', 
                                       'super().__init__(agent_id, config)', content)
                        
                        # Remove methods that don't exist in BaseAgent
                        content = re.sub(r'def _register_capabilities\(self\):.*?(?=\n    def|\n\nclass|\Z)', '', 
                                       content, flags=re.DOTALL)
                        content = re.sub(r'def _register_message_handlers\(self\):.*?(?=\n    def|\n\nclass|\Z)', '', 
                                       content, flags=re.DOTALL)
                    
                    # Write back if changed
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Fixed imports in: {filepath}")
                
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    print("\nImport fixes completed!")

def fix_test_imports():
    """Fix test file imports"""
    print("\nFixing test imports...")
    
    test_replacements = [
        # Fix RSI agent import
        (r'from agents\.technical\.rsi_agent import RSIAgent', 
         'from agents.core.technical.momentum.rsi_agent import RSIAgent'),
    ]
    
    # Fix test files
    test_files = [
        'tests/agents/test_rsi_agent.py',
        'tests/test_rsi_agent.py',
    ]
    
    for filepath in test_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                for pattern, replacement in test_replacements:
                    content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed test imports in: {filepath}")
            
            except Exception as e:
                print(f"Error fixing {filepath}: {e}")

if __name__ == "__main__":
    fix_agent_imports()
    fix_test_imports() 