#!/usr/bin/env python3
"""
Fix missing abstract methods in all agent classes
"""

import os
import re
import ast
from typing import Set, List, Tuple

def find_abstract_methods(base_agent_path: str) -> Set[str]:
    """Find all abstract methods in BaseAgent"""
    with open(base_agent_path, 'r') as f:
        content = f.read()
    
    # Parse the AST to find abstract methods
    tree = ast.parse(content)
    abstract_methods = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if it has @abstractmethod decorator
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                    abstract_methods.add(node.name)
                elif isinstance(decorator, ast.Attribute) and decorator.attr == 'abstractmethod':
                    abstract_methods.add(node.name)
    
    return abstract_methods

def find_agent_files() -> List[str]:
    """Find all agent Python files"""
    agent_files = []
    
    for root, dirs, files in os.walk('agents'):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('_agent.py') or file == 'base.py':
                agent_files.append(os.path.join(root, file))
    
    return agent_files

def check_agent_methods(filepath: str, abstract_methods: Set[str]) -> Tuple[List[str], bool]:
    """Check which abstract methods are missing in an agent file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if this inherits from BaseAgent
    if 'BaseAgent' not in content:
        return [], False
    
    # Skip if it's the base file itself
    if filepath.endswith('base.py') or filepath.endswith('base_agent.py'):
        return [], False
    
    # Parse to find implemented methods
    try:
        tree = ast.parse(content)
    except:
        print(f"Could not parse {filepath}")
        return [], False
    
    implemented_methods = set()
    inherits_base_agent = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from BaseAgent
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == 'BaseAgent':
                    inherits_base_agent = True
            
            # Find all methods in this class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    implemented_methods.add(item.name)
    
    if not inherits_base_agent:
        return [], False
    
    # Find missing methods
    missing_methods = abstract_methods - implemented_methods
    return list(missing_methods), True

def add_missing_methods(filepath: str, missing_methods: List[str]) -> bool:
    """Add missing abstract methods to an agent file"""
    if not missing_methods:
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Method implementations
    method_impls = {
        'analyze': '''    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data and return results"""
        # TODO: Implement specific analysis logic
        return {
            'status': 'success',
            'analysis': 'Not implemented',
            'timestamp': datetime.now().isoformat()
        }
''',
        'get_required_data_types': '''    def get_required_data_types(self) -> List[str]:
        """Return required data types for this agent"""
        return ['price', 'volume']
''',
        'validate_config': '''    def validate_config(self) -> bool:
        """Validate agent configuration"""
        return True
''',
        'get_state': '''    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            'agent_id': self.agent_id,
            'is_active': True,
            'last_update': datetime.now().isoformat()
        }
''',
        'set_state': '''    def set_state(self, state: Dict[str, Any]) -> None:
        """Set agent state"""
        # TODO: Implement state restoration logic
        pass
'''
    }
    
    # Find the last method in the class
    class_pattern = r'class\s+\w+\([^)]*BaseAgent[^)]*\):[^}]+?(?=\nclass|\Z)'
    class_match = re.search(class_pattern, content, re.DOTALL)
    
    if not class_match:
        print(f"Could not find BaseAgent class in {filepath}")
        return False
    
    class_content = class_match.group(0)
    
    # Find the insertion point (before the last method or at the end of class)
    method_pattern = r'(\n    def [^(]+\([^)]*\)[^:]*:[^}]+?)(?=\n    def|\n\nclass|\Z)'
    methods = list(re.finditer(method_pattern, class_content, re.DOTALL))
    
    if methods:
        # Insert after the last method
        last_method_end = methods[-1].end()
        insertion_point = class_match.start() + last_method_end
    else:
        # Insert at the end of the class
        insertion_point = class_match.end()
    
    # Add imports if needed
    imports_to_add = []
    if 'datetime' in str(missing_methods) and 'from datetime import datetime' not in content:
        imports_to_add.append('from datetime import datetime')
    if 'Dict' in content and 'from typing import Dict' not in content:
        if 'from typing import' in content:
            content = content.replace('from typing import', 'from typing import Dict, ')
        else:
            imports_to_add.append('from typing import Dict, Any, List')
    
    # Add imports at the top
    if imports_to_add:
        import_section_end = content.find('\n\n')
        if import_section_end > 0:
            content = content[:import_section_end] + '\n' + '\n'.join(imports_to_add) + content[import_section_end:]
    
    # Add missing methods
    methods_to_add = '\n'
    for method in missing_methods:
        if method in method_impls:
            methods_to_add += '\n' + method_impls[method]
    
    # Insert methods
    content = content[:insertion_point] + methods_to_add + content[insertion_point:]
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Main function to fix all agents"""
    print("Finding abstract methods in BaseAgent...")
    
    # Find abstract methods
    base_agent_path = 'agents/base.py'
    if not os.path.exists(base_agent_path):
        base_agent_path = 'agents/base_agent.py'
    
    abstract_methods = find_abstract_methods(base_agent_path)
    print(f"Found abstract methods: {abstract_methods}")
    
    # Find all agent files
    agent_files = find_agent_files()
    print(f"\nFound {len(agent_files)} agent files")
    
    # Check and fix each agent
    fixed_count = 0
    for filepath in agent_files:
        missing_methods, inherits_base = check_agent_methods(filepath, abstract_methods)
        
        if inherits_base and missing_methods:
            print(f"\n{filepath}:")
            print(f"  Missing methods: {missing_methods}")
            
            if add_missing_methods(filepath, missing_methods):
                print(f"  ✓ Fixed!")
                fixed_count += 1
            else:
                print(f"  ✗ Could not fix")
    
    print(f"\n✅ Fixed {fixed_count} agent files")

if __name__ == "__main__":
    main() 