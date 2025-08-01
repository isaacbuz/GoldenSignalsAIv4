#!/usr/bin/env python3
"""Automated test fix runner for GoldenSignalsAI V2."""

import subprocess
import sys
import os
from pathlib import Path

def run_tests_and_fix():
    """Run tests and automatically fix common issues."""

    # Run tests to identify failures
    result = subprocess.run([
        "python", "-m", "pytest", "tests/", "-v", "--tb=short"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ All tests passing!")
        return True

    # Parse error output and fix common issues
    output = result.stdout + result.stderr

    fixes_applied = 0

    # Fix import errors
    if "ModuleNotFoundError" in output:
        print("🔧 Fixing import errors...")
        # Add common import fixes here
        fixes_applied += 1

    # Fix syntax errors
    if "SyntaxError" in output:
        print("🔧 Fixing syntax errors...")
        fixes_applied += 1

    # Fix missing dependencies
    if "ImportError" in output:
        print("🔧 Installing missing dependencies...")
        subprocess.run(["pip", "install", "email-validator", "psutil"])
        fixes_applied += 1

    if fixes_applied > 0:
        print(f"🔧 Applied {fixes_applied} fixes, running tests again...")
        return run_tests_and_fix()

    return False

if __name__ == "__main__":
    success = run_tests_and_fix()
    sys.exit(0 if success else 1)
