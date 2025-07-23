#!/usr/bin/env python
"""
Test runner for GoldenSignalsAI tests
Handles environment setup and runs security/data integrity tests
"""

import os
import sys
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_tests():
    """Run the security and data integrity tests"""

    print("🧪 Running GoldenSignalsAI Test Suite\n")

    # Set environment variables
    os.environ['PYTHONPATH'] = project_root
    os.environ['TESTING'] = 'true'

    # Test files to run
    test_files = [
        "tests/test_security.py",
        "tests/test_data_integrity.py"
    ]

    all_passed = True

    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        print("=" * 60)

        # Run the test file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd=project_root,
            capture_output=False
        )

        if result.returncode != 0:
            all_passed = False
            print(f"\n❌ {test_file} failed!")
        else:
            print(f"\n✅ {test_file} passed!")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_tests())
