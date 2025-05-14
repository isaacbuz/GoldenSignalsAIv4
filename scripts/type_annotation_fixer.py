import ast
import astor
import os
import re
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

class TypeAnnotationFixer(ast.NodeTransformer):
    def __init__(self):
        self.type_map = {
            'list': 'List[Any]',
            'dict': 'Dict[str, Any]',
            'pd.DataFrame': 'pd.DataFrame',
            'np.ndarray': 'np.ndarray',
            'int': 'int',
            'float': 'float',
            'str': 'str',
            'bool': 'bool'
        }

    def _infer_type(self, node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return self.type_map.get(node.func.id, 'Any')
            elif isinstance(node.func, ast.Attribute):
                return self.type_map.get(f'{node.func.value.id}.{node.func.attr}', 'Any')
        
        if isinstance(node, ast.List):
            return 'List[Any]'
        if isinstance(node, ast.Dict):
            return 'Dict[str, Any]'
        if isinstance(node, ast.Num):
            return 'float' if isinstance(node.n, float) else 'int'
        if isinstance(node, ast.Str):
            return 'str'
        if isinstance(node, ast.NameConstant):
            return type(node.value).__name__
        
        return 'Any'
    def visit_FunctionDef(self, node):
        # Sophisticated return type inference
        if not node.returns:
            # Try to infer return type from function body
            return_type = self._infer_return_type(node)
            node.returns = ast.Name(id=return_type, ctx=ast.Load())
        
        # Enhanced argument type annotation
        for arg in node.args.args:
            if not arg.annotation:
                # Try to infer type based on function usage
                inferred_type = self._infer_argument_type(node, arg.arg)
                arg.annotation = ast.Name(id=inferred_type, ctx=ast.Load())
        
        return self.generic_visit(node)

    def _infer_return_type(self, node):
        # Analyze function body to infer return type
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                if stmt.value:
                    return self._infer_type(stmt.value)
        return 'Any'

    def _infer_argument_type(self, node, arg_name):
        # Analyze function body to infer argument type
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == arg_name:
                        return self._infer_type(stmt.value)
        return 'Any'

    def visit_Assign(self, node):
        # Add type hints to variable assignments
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            if not hasattr(target, 'annotation'):
                target.annotation = ast.Name(id='Any', ctx=ast.Load())
        
        return self.generic_visit(node)

def fix_type_annotations(file_path):
    with open(file_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    transformer = TypeAnnotationFixer()
    modified_tree = transformer.visit(tree)
    
    # Add necessary imports
    imports_to_add = [
        'from typing import Dict, List, Any, Optional, Union',
        'import pandas as pd'
    ]
    
    modified_source = astor.to_source(modified_tree)
    
    # Prepend imports
    modified_source = '\n'.join(imports_to_add) + '\n\n' + modified_source
    
    with open(file_path, 'w') as f:
        f.write(modified_source)

def process_directory(directory, exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = [
            '__init__.py', 
            'test_', 
            'conftest.py', 
            'setup.py'
        ]
    
    processed_files = []
    skipped_files = []
    error_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Skip excluded files
                if any(pattern in file for pattern in exclude_patterns):
                    skipped_files.append(file_path)
                    continue
                
                try:
                    fix_type_annotations(file_path)
                    processed_files.append(file_path)
                    print(f"Processed: {file_path}")
                except Exception as e:
                    error_files.append((file_path, str(e)))
                    print(f"Error processing {file_path}: {e}")
    
    # Generate summary report
    print("\n--- Type Annotation Fixer Summary ---")
    print(f"Processed Files: {len(processed_files)}")
    print(f"Skipped Files: {len(skipped_files)}")
    print(f"Error Files: {len(error_files)}")
    
    if error_files:
        print("\nFiles with Errors:")
        for file, error in error_files:
            print(f"- {file}: {error}")
    
    return {
        'processed': processed_files,
        'skipped': skipped_files,
        'errors': error_files
    }

if __name__ == '__main__':
    project_root = '/Users/isaacbuz/Documents/Projects/GoldenSignalsAI/goldensignalsai'
    process_directory(project_root)
