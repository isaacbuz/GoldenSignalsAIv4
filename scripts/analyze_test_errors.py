#!/usr/bin/env python3
"""
Analyze remaining test collection errors and categorize them.
"""

import subprocess
import re
from collections import defaultdict

def get_test_errors():
    """Run pytest and capture all error details."""
    result = subprocess.run(
        ["python", "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True
    )

    errors = defaultdict(list)
    current_file = None
    current_error = []

    lines = result.stderr.split('\n')

    for i, line in enumerate(lines):
        if "ERROR collecting" in line:
            # Save previous error if exists
            if current_file and current_error:
                error_msg = '\n'.join(current_error).strip()
                errors[categorize_error(error_msg)].append((current_file, error_msg))

            # Start new error
            match = re.search(r'ERROR collecting (.+\.py)', line)
            if match:
                current_file = match.group(1)
                current_error = []
        elif current_file and line.strip():
            current_error.append(line)

    # Don't forget the last error
    if current_file and current_error:
        error_msg = '\n'.join(current_error).strip()
        errors[categorize_error(error_msg)].append((current_file, error_msg))

    return errors

def categorize_error(error_msg):
    """Categorize error based on the error message."""
    if "ModuleNotFoundError" in error_msg:
        if "No module named" in error_msg:
            match = re.search(r"No module named '([^']+)'", error_msg)
            if match:
                module = match.group(1)
                if module.startswith('agents.'):
                    return f"Missing agent module: {module}"
                elif module in ['websocket', 'transformers', 'torch', 'tensorflow']:
                    return f"Missing dependency: {module}"
                else:
                    return f"Missing module: {module}"
    elif "ImportError" in error_msg:
        if "cannot import name" in error_msg:
            match = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_msg)
            if match:
                return f"Import error: {match.group(1)} from {match.group(2)}"
    elif "AttributeError" in error_msg:
        return "Attribute error"
    elif "SyntaxError" in error_msg:
        return "Syntax error"

    return "Other error"

def suggest_fixes(errors):
    """Suggest fixes for common error patterns."""
    fixes = []

    for category, file_errors in errors.items():
        if category.startswith("Missing dependency:"):
            module = category.split(": ")[1]
            if module == "websocket":
                fixes.append("pip install websocket-client")
            elif module == "transformers":
                fixes.append("pip install transformers")
            elif module == "torch":
                fixes.append("pip install torch")
            elif module == "tensorflow":
                fixes.append("pip install tensorflow")

        elif category.startswith("Missing agent module:"):
            module = category.split(": ")[1]
            fixes.append(f"# Check if {module} exists or update import path")

        elif category.startswith("Import error:"):
            fixes.append(f"# Fix import for {category}")

    return fixes

def main():
    """Analyze test errors and provide summary."""
    print("Analyzing test collection errors...\n")

    errors = get_test_errors()

    # Print summary
    total_errors = sum(len(files) for files in errors.values())
    print(f"Total errors: {total_errors}")
    print(f"Error categories: {len(errors)}\n")

    # Print errors by category
    for category, file_errors in sorted(errors.items()):
        print(f"\n{category} ({len(file_errors)} files):")
        for file, error in file_errors[:3]:  # Show first 3
            print(f"  - {file}")
            if len(error.split('\n')) > 1:
                print(f"    {error.split(chr(10))[-1][:60]}...")
        if len(file_errors) > 3:
            print(f"  ... and {len(file_errors) - 3} more")

    # Suggest fixes
    print("\n\nSuggested fixes:")
    fixes = suggest_fixes(errors)
    for fix in fixes:
        print(f"  {fix}")

    # List files that need manual review
    print("\n\nFiles needing manual review:")
    manual_review = []
    for category, file_errors in errors.items():
        if "Other error" in category or "Attribute error" in category:
            manual_review.extend([f for f, _ in file_errors])

    for file in sorted(set(manual_review))[:10]:
        print(f"  - {file}")

if __name__ == "__main__":
    main()
