#!/usr/bin/env python3
"""
Comprehensive Test Runner for GoldenSignalsAI V2
Orchestrates all tests across the codebase with detailed logging and reporting
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Setup paths
PROJECT_ROOT = Path(__file__).parent.absolute()
TEST_LOG_DIR = PROJECT_ROOT / "test_logs"
TEST_LOG_DIR.mkdir(exist_ok=True)

# Configure logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = TEST_LOG_DIR / f"test_run_{timestamp}.log"

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# Console handler with colors
class ColoredConsoleHandler(logging.StreamHandler):
    """Console handler with colored output"""

    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;37m',     # White
        'WARNING': '\033[1;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[1;31m', # Bold Red
        'SUCCESS': '\033[0;32m',  # Green
        'RESET': '\033[0m'
    }

    def emit(self, record):
        try:
            msg = self.format(record)
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])

            # Add custom SUCCESS level handling
            if hasattr(record, 'success') and record.success:
                color = self.COLORS['SUCCESS']

            self.stream.write(f"{color}{msg}{self.COLORS['RESET']}\n")
            self.flush()
        except Exception:
            self.handleError(record)

console_handler = ColoredConsoleHandler()
console_handler.setFormatter(formatter)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Add custom SUCCESS level
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        kwargs['extra'] = {'success': True}
        self._log(logging.SUCCESS, message, args, **kwargs)

logging.Logger.success = success


class TestStatus(Enum):
    """Test execution status"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


@dataclass
class TestResult:
    """Test execution result"""
    module: str
    test_type: str
    status: TestStatus = TestStatus.NOT_RUN
    duration: float = 0.0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    output: str = ""
    error_details: str = ""
    command: List[str] = field(default_factory=list)


class TestRunner:
    """Master test runner for all modules"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        self.summary_file = TEST_LOG_DIR / f"test_summary_{timestamp}.json"

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")

        checks = {
            "Python 3": self._check_python(),
            "Node.js": self._check_node(),
            "Virtual Environment": self._check_venv(),
            "Python Dependencies": self._check_python_deps(),
            "Frontend Dependencies": self._check_frontend_deps()
        }

        all_passed = True
        for check, passed in checks.items():
            if passed:
                logger.success(f"✓ {check}")
            else:
                logger.error(f"✗ {check}")
                all_passed = False

        return all_passed

    def _check_python(self) -> bool:
        """Check Python installation"""
        try:
            result = subprocess.run(
                [sys.executable, "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_node(self) -> bool:
        """Check Node.js installation"""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_venv(self) -> bool:
        """Check if running in virtual environment"""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

    def _check_python_deps(self) -> bool:
        """Check if Python dependencies are installed"""
        try:
            import pytest
            import pandas
            import numpy
            return True
        except ImportError:
            return False

    def _check_frontend_deps(self) -> bool:
        """Check if frontend dependencies are installed"""
        frontend_dir = PROJECT_ROOT / "frontend"
        node_modules = frontend_dir / "node_modules"
        return node_modules.exists()

    def run_command(self, command: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
        """Run a command and capture output"""
        try:
            logger.debug(f"Running command: {' '.join(command)}")

            if env is None:
                env = os.environ.copy()
            # Ensure Python path includes project root
            env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

            result = subprocess.run(
                command,
                cwd=cwd or PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return -1, "", "Command timed out after 5 minutes"
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return -1, "", str(e)

    def parse_pytest_output(self, output: str, error: str = "") -> Dict[str, int]:
        """Parse pytest output for test statistics"""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Combine stdout and stderr
        full_output = output + "\n" + error

        # Look for pytest summary line
        for line in full_output.split('\n'):
            # Pattern: "X passed, Y failed, Z skipped"
            if " passed" in line or " failed" in line or " skipped" in line or " error" in line:
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if " passed" in part:
                        try:
                            num = part.split()[0]
                            stats["passed"] = int(num)
                        except (ValueError, IndexError):
                            pass
                    elif " failed" in part:
                        try:
                            num = part.split()[0]
                            stats["failed"] = int(num)
                        except (ValueError, IndexError):
                            pass
                    elif " skipped" in part:
                        try:
                            num = part.split()[0]
                            stats["skipped"] = int(num)
                        except (ValueError, IndexError):
                            pass
                    elif " error" in part:
                        try:
                            num = part.split()[0]
                            stats["errors"] = int(num)
                        except (ValueError, IndexError):
                            pass

        return stats

    def parse_npm_test_output(self, output: str, error: str = "") -> Dict[str, int]:
        """Parse npm test output for test statistics"""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Combine stdout and stderr
        full_output = output + "\n" + error

        # Look for test summary patterns
        for line in full_output.split('\n'):
            # Vitest format
            if "Tests:" in line or "Test Suites:" in line:
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        try:
                            num = part.split()[0]
                            stats["passed"] += int(num)
                        except (ValueError, IndexError):
                            pass
                    elif "failed" in part:
                        try:
                            num = part.split()[0]
                            stats["failed"] += int(num)
                        except (ValueError, IndexError):
                            pass
                    elif "skipped" in part:
                        try:
                            num = part.split()[0]
                            stats["skipped"] += int(num)
                        except (ValueError, IndexError):
                            pass
            # Count individual test markers
            elif "✓" in line:
                stats["passed"] += line.count("✓")
            elif "✗" in line or "×" in line:
                stats["failed"] += line.count("✗") + line.count("×")

        return stats

    def run_test_suite(self, result: TestResult) -> TestResult:
        """Run a single test suite"""
        logger.info("=" * 80)
        logger.info(f"Running {result.module} - {result.test_type}")
        logger.info(f"Command: {' '.join(result.command)}")
        logger.info("=" * 80)

        start_time = time.time()

        # Determine working directory
        cwd = PROJECT_ROOT
        if result.module == "Frontend" and (PROJECT_ROOT / "frontend").exists():
            cwd = PROJECT_ROOT / "frontend"

        # Set TEST_MODE for comprehensive system test
        env = None
        if "test_comprehensive_system.py" in ' '.join(result.command):
            env = os.environ.copy()
            env["TEST_MODE"] = "true"

        # Run the test
        returncode, stdout, stderr = self.run_command(result.command, cwd=cwd, env=env)

        result.duration = time.time() - start_time
        result.output = stdout
        result.error_details = stderr

        # Parse results based on test type
        if "pytest" in result.command[0] or "pytest" in str(result.command):
            stats = self.parse_pytest_output(stdout, stderr)
        elif "npm" in result.command[0]:
            stats = self.parse_npm_test_output(stdout, stderr)
        else:
            # For other commands, simple pass/fail
            stats = {
                "passed": 1 if returncode == 0 else 0,
                "failed": 1 if returncode != 0 else 0,
                "skipped": 0,
                "errors": 0
            }

        result.passed = stats["passed"]
        result.failed = stats["failed"]
        result.skipped = stats["skipped"]
        result.errors = stats["errors"]

        # Determine status
        if returncode == 0 and result.failed == 0:
            result.status = TestStatus.PASSED
            logger.success(f"✓ {result.test_type} PASSED ({result.duration:.2f}s)")
        else:
            result.status = TestStatus.FAILED
            logger.error(f"✗ {result.test_type} FAILED ({result.duration:.2f}s)")

            # Log first few lines of error
            if stderr:
                error_lines = stderr.split('\n')[:10]
                for line in error_lines:
                    if line.strip():
                        logger.error(f"  {line}")

        return result

    def run_backend_tests(self):
        """Run all backend Python tests"""
        logger.info("\n" + "=" * 80)
        logger.info("BACKEND TESTS")
        logger.info("=" * 80)

        test_suites = [
            TestResult(
                module="Backend",
                test_type="Unit Tests",
                command=[sys.executable, "-m", "pytest", "tests/unit", "-v", "--tb=short"]
            ),
            TestResult(
                module="Backend",
                test_type="Integration Tests",
                command=[sys.executable, "-m", "pytest", "tests/integration", "-v", "--tb=short"]
            ),
            TestResult(
                module="Backend",
                test_type="Agent Tests",
                command=[sys.executable, "-m", "pytest", "tests/agents", "-v", "--tb=short"]
            ),
            TestResult(
                module="Backend",
                test_type="Performance Tests",
                command=[sys.executable, "-m", "pytest", "tests/performance", "-v", "--tb=short", "-m", "not slow"]
            ),
            TestResult(
                module="Backend",
                test_type="Comprehensive System Test",
                command=[sys.executable, "tests/test_comprehensive_system.py"]
            )
        ]

        for test_suite in test_suites:
            result = self.run_test_suite(test_suite)
            self.results.append(result)

    def run_frontend_tests(self):
        """Run all frontend tests"""
        logger.info("\n" + "=" * 80)
        logger.info("FRONTEND TESTS")
        logger.info("=" * 80)

        frontend_dir = PROJECT_ROOT / "frontend"
        if not frontend_dir.exists():
            logger.warning("Frontend directory not found, skipping frontend tests")
            return

        test_suites = [
            TestResult(
                module="Frontend",
                test_type="Unit & Integration Tests",
                command=["npm", "test", "--", "--run"]
            ),
            TestResult(
                module="Frontend",
                test_type="Component Tests",
                command=["npm", "test", "--", "--run", "src/components"]
            )
        ]

        # Add E2E tests if Cypress is configured
        if (frontend_dir / "cypress.config.ts").exists():
            test_suites.append(
                TestResult(
                    module="Frontend",
                    test_type="E2E Tests",
                    command=["npm", "run", "test:e2e:headless"]
                )
            )

        for test_suite in test_suites:
            result = self.run_test_suite(test_suite)
            self.results.append(result)

    def run_ml_tests(self):
        """Run ML model tests"""
        logger.info("\n" + "=" * 80)
        logger.info("ML MODEL TESTS")
        logger.info("=" * 80)

        test_suites = []

        # ML Models tests
        if (PROJECT_ROOT / "ml_models" / "tests").exists():
            test_suites.append(
                TestResult(
                    module="ML",
                    test_type="ML Model Tests",
                    command=[sys.executable, "-m", "pytest", "ml_models/tests", "-v", "--tb=short"]
                )
            )

        # ML Training tests
        if (PROJECT_ROOT / "ml_training").exists():
            test_suites.append(
                TestResult(
                    module="ML",
                    test_type="ML Training Tests",
                    command=[sys.executable, "-m", "pytest", "ml_training", "-v", "--tb=short", "-k", "test"]
                )
            )

        for test_suite in test_suites:
            result = self.run_test_suite(test_suite)
            self.results.append(result)

    def run_infrastructure_tests(self):
        """Run infrastructure and configuration tests"""
        logger.info("\n" + "=" * 80)
        logger.info("INFRASTRUCTURE TESTS")
        logger.info("=" * 80)

        test_suites = []

        # Config validation
        if (PROJECT_ROOT / "config.yaml").exists():
            test_suites.append(
                TestResult(
                    module="Infrastructure",
                    test_type="Config Validation",
                    command=[sys.executable, "-c", "import yaml; yaml.safe_load(open('config.yaml')); print('Config is valid')"]
                )
            )

        # Database connection test
        if (PROJECT_ROOT / "check_databases.py").exists():
            test_suites.append(
                TestResult(
                    module="Infrastructure",
                    test_type="Database Connection Test",
                    command=[sys.executable, "check_databases.py"]
                )
            )

        for test_suite in test_suites:
            result = self.run_test_suite(test_suite)
            self.results.append(result)

    def generate_report(self) -> bool:
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST EXECUTION SUMMARY")
        logger.info("=" * 80)

        # Calculate totals
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        # Group by module
        modules = {}
        for result in self.results:
            if result.module not in modules:
                modules[result.module] = []
            modules[result.module].append(result)

        # Print summary by module
        for module, results in modules.items():
            logger.info(f"\n{module} Tests:")
            for result in results:
                status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
                status_color = "\033[0;32m" if result.status == TestStatus.PASSED else "\033[0;31m"
                reset_color = "\033[0m"

                logger.info(
                    f"  {status_color}{status_symbol}{reset_color} {result.test_type}: "
                    f"{result.passed} passed, {result.failed} failed, "
                    f"{result.skipped} skipped ({result.duration:.2f}s)"
                )

        # Overall summary
        logger.info("\n" + "-" * 80)
        logger.info("Overall Statistics:")
        logger.info(f"  Total Tests Passed: {total_passed}")
        logger.info(f"  Total Tests Failed: {total_failed}")
        logger.info(f"  Total Tests Skipped: {total_skipped}")
        logger.info(f"  Total Errors: {total_errors}")
        logger.info(f"  Total Duration: {total_duration:.2f}s")

        success_rate = 0
        if (total_passed + total_failed) > 0:
            success_rate = (total_passed / (total_passed + total_failed)) * 100
        logger.info(f"  Success Rate: {success_rate:.2f}%")

        # Save detailed report
        report_data = {
            "execution_time": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration": total_duration
            },
            "summary": {
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_errors": total_errors,
                "success_rate": success_rate
            },
            "results": [
                {
                    "module": r.module,
                    "test_type": r.test_type,
                    "status": r.status.value,
                    "duration": r.duration,
                    "passed": r.passed,
                    "failed": r.failed,
                    "skipped": r.skipped,
                    "errors": r.errors,
                    "command": r.command
                }
                for r in self.results
            ]
        }

        with open(self.summary_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"\nDetailed report saved to: {self.summary_file}")
        logger.info(f"Full log available at: {log_file}")

        # Print failed tests details
        failed_tests = [r for r in self.results if r.status == TestStatus.FAILED]
        if failed_tests:
            logger.error("\n" + "=" * 80)
            logger.error("FAILED TESTS DETAILS")
            logger.error("=" * 80)

            for result in failed_tests:
                logger.error(f"\n{result.module} - {result.test_type}:")
                logger.error(f"Command: {' '.join(result.command)}")
                if result.error_details:
                    logger.error("Error output (first 20 lines):")
                    error_lines = result.error_details.split('\n')[:20]
                    for line in error_lines:
                        if line.strip():
                            logger.error(f"  {line}")

        all_passed = total_failed == 0 and total_errors == 0

        if all_passed:
            logger.success("\n" + "=" * 80)
            logger.success("ALL TESTS PASSED! 🎉")
            logger.success("=" * 80)
        else:
            logger.error("\n" + "=" * 80)
            logger.error("SOME TESTS FAILED")
            logger.error("=" * 80)

        return all_passed

    def run_all_tests(self, modules: Optional[List[str]] = None):
        """Run all tests or specific modules"""
        self.start_time = datetime.now()
        logger.info(f"Starting test execution at {self.start_time}")

        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Please install missing dependencies.")
            return False

        # Determine which test suites to run
        if modules:
            logger.info(f"Running specific modules: {', '.join(modules)}")
            if "backend" in modules:
                self.run_backend_tests()
            if "frontend" in modules:
                self.run_frontend_tests()
            if "ml" in modules:
                self.run_ml_tests()
            if "infrastructure" in modules:
                self.run_infrastructure_tests()
        else:
            # Run all test suites
            self.run_backend_tests()
            self.run_frontend_tests()
            self.run_ml_tests()
            self.run_infrastructure_tests()

        self.end_time = datetime.now()
        success = self.generate_report()

        return success


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for GoldenSignalsAI V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                    # Run all tests
  python test_runner.py --module backend   # Run only backend tests
  python test_runner.py --module frontend  # Run only frontend tests
  python test_runner.py --list             # List all available test suites
        """
    )

    parser.add_argument(
        "--module",
        nargs="+",
        choices=["backend", "frontend", "ml", "infrastructure"],
        help="Run tests for specific modules only"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test suites"
    )

    args = parser.parse_args()

    if args.list:
        print("Available test modules:")
        print("  - backend: Backend unit, integration, and system tests")
        print("  - frontend: Frontend unit, component, and E2E tests")
        print("  - ml: Machine learning model and training tests")
        print("  - infrastructure: Configuration and infrastructure tests")
        sys.exit(0)

    runner = TestRunner()

    try:
        success = runner.run_all_tests(modules=args.module)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.error("\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
