#!/usr/bin/env python3
"""
Master Test Runner for GoldenSignalsAI V2
Runs all tests across the entire codebase with comprehensive logging and reporting
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
import concurrent.futures
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
LOG_DIR = Path("test_logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOG_DIR / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Test execution result"""
    module: str
    test_type: str
    status: TestStatus
    duration: float
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    output: str = ""
    error_details: str = ""


class TestRunner:
    """Master test runner for all modules"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and capture output"""
        logger.info(f"Running command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return -1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return -1, "", str(e)

    def parse_pytest_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output for test statistics"""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Look for pytest summary line
        for line in output.split('\n'):
            if "passed" in line or "failed" in line or "skipped" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part and i > 0:
                        try:
                            stats["passed"] = int(parts[i-1])
                        except ValueError:
                            pass
                    elif "failed" in part and i > 0:
                        try:
                            stats["failed"] = int(parts[i-1])
                        except ValueError:
                            pass
                    elif "skipped" in part and i > 0:
                        try:
                            stats["skipped"] = int(parts[i-1])
                        except ValueError:
                            pass
                    elif "error" in part and i > 0:
                        try:
                            stats["errors"] = int(parts[i-1])
                        except ValueError:
                            pass

        return stats

    def parse_npm_test_output(self, output: str) -> Dict[str, int]:
        """Parse npm test output for test statistics"""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Look for test summary patterns
        for line in output.split('\n'):
            if "Tests:" in line:
                # Vitest format: Tests: X passed, Y failed, Z skipped
                parts = line.replace("Tests:", "").strip().split(",")
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        try:
                            stats["passed"] = int(part.split()[0])
                        except ValueError:
                            pass
                    elif "failed" in part:
                        try:
                            stats["failed"] = int(part.split()[0])
                        except ValueError:
                            pass
                    elif "skipped" in part:
                        try:
                            stats["skipped"] = int(part.split()[0])
                        except ValueError:
                            pass
            elif "✓" in line:
                stats["passed"] += 1
            elif "✗" in line or "×" in line:
                stats["failed"] += 1

        return stats

    def run_backend_tests(self):
        """Run all backend Python tests"""
        logger.info("=" * 80)
        logger.info("Running Backend Tests")
        logger.info("=" * 80)

        test_modules = [
            ("Unit Tests", ["pytest", "tests/unit", "-v", "--tb=short"]),
            ("Integration Tests", ["pytest", "tests/integration", "-v", "--tb=short"]),
            ("Agent Tests", ["pytest", "tests/agents", "-v", "--tb=short"]),
            ("Performance Tests", ["pytest", "tests/performance", "-v", "--tb=short", "-m", "not slow"]),
            ("Root Tests", ["pytest", "tests/root_tests", "-v", "--tb=short"]),
            ("Comprehensive System Test", ["python", "tests/test_comprehensive_system.py"])
        ]

        for test_name, command in test_modules:
            logger.info(f"\nRunning {test_name}...")
            start_time = time.time()

            returncode, stdout, stderr = self.run_command(command)
            duration = time.time() - start_time

            # Parse results
            if "pytest" in command[0]:
                stats = self.parse_pytest_output(stdout + stderr)
            else:
                # For direct python scripts
                stats = {"passed": 1 if returncode == 0 else 0, "failed": 1 if returncode != 0 else 0}

            result = TestResult(
                module="Backend",
                test_type=test_name,
                status=TestStatus.PASSED if returncode == 0 else TestStatus.FAILED,
                duration=duration,
                passed=stats.get("passed", 0),
                failed=stats.get("failed", 0),
                skipped=stats.get("skipped", 0),
                errors=stats.get("errors", 0),
                output=stdout,
                error_details=stderr if returncode != 0 else ""
            )

            self.results.append(result)
            logger.info(f"{test_name} completed in {duration:.2f}s - Status: {result.status.value}")

    def run_frontend_tests(self):
        """Run all frontend tests"""
        logger.info("=" * 80)
        logger.info("Running Frontend Tests")
        logger.info("=" * 80)

        frontend_dir = Path("frontend")

        # Check if frontend directory exists
        if not frontend_dir.exists():
            logger.error("Frontend directory not found")
            return

        test_commands = [
            ("Unit & Integration Tests", ["npm", "test", "--", "--run"]),
            ("Component Tests", ["npm", "test", "--", "--run", "src/components"]),
            ("Hook Tests", ["npm", "test", "--", "--run", "src/hooks"]),
            ("E2E Tests", ["npm", "run", "test:e2e"])
        ]

        for test_name, command in test_commands:
            logger.info(f"\nRunning {test_name}...")
            start_time = time.time()

            returncode, stdout, stderr = self.run_command(command, cwd=frontend_dir)
            duration = time.time() - start_time

            # Parse results
            stats = self.parse_npm_test_output(stdout + stderr)

            result = TestResult(
                module="Frontend",
                test_type=test_name,
                status=TestStatus.PASSED if returncode == 0 else TestStatus.FAILED,
                duration=duration,
                passed=stats.get("passed", 0),
                failed=stats.get("failed", 0),
                skipped=stats.get("skipped", 0),
                errors=stats.get("errors", 0),
                output=stdout,
                error_details=stderr if returncode != 0 else ""
            )

            self.results.append(result)
            logger.info(f"{test_name} completed in {duration:.2f}s - Status: {result.status.value}")

    def run_ml_tests(self):
        """Run ML model tests"""
        logger.info("=" * 80)
        logger.info("Running ML Model Tests")
        logger.info("=" * 80)

        test_commands = [
            ("ML Model Tests", ["pytest", "ml_models/tests", "-v", "--tb=short"]),
            ("ML Training Tests", ["pytest", "ml_training", "-v", "--tb=short", "-k", "test"]),
            ("Research ML Tests", ["pytest", "tests/agents/research", "-v", "--tb=short"])
        ]

        for test_name, command in test_commands:
            logger.info(f"\nRunning {test_name}...")
            start_time = time.time()

            returncode, stdout, stderr = self.run_command(command)
            duration = time.time() - start_time

            stats = self.parse_pytest_output(stdout + stderr)

            result = TestResult(
                module="ML",
                test_type=test_name,
                status=TestStatus.PASSED if returncode == 0 else TestStatus.FAILED,
                duration=duration,
                passed=stats.get("passed", 0),
                failed=stats.get("failed", 0),
                skipped=stats.get("skipped", 0),
                errors=stats.get("errors", 0),
                output=stdout,
                error_details=stderr if returncode != 0 else ""
            )

            self.results.append(result)
            logger.info(f"{test_name} completed in {duration:.2f}s - Status: {result.status.value}")

    def run_infrastructure_tests(self):
        """Run infrastructure and deployment tests"""
        logger.info("=" * 80)
        logger.info("Running Infrastructure Tests")
        logger.info("=" * 80)

        test_commands = [
            ("Docker Build Test", ["docker", "build", "-t", "goldensignals-test", "."]),
            ("Config Validation", ["python", "-m", "src.config", "--validate"]),
            ("Database Connection Test", ["python", "check_databases.py"])
        ]

        for test_name, command in test_commands:
            logger.info(f"\nRunning {test_name}...")
            start_time = time.time()

            returncode, stdout, stderr = self.run_command(command)
            duration = time.time() - start_time

            result = TestResult(
                module="Infrastructure",
                test_type=test_name,
                status=TestStatus.PASSED if returncode == 0 else TestStatus.FAILED,
                duration=duration,
                passed=1 if returncode == 0 else 0,
                failed=1 if returncode != 0 else 0,
                output=stdout,
                error_details=stderr if returncode != 0 else ""
            )

            self.results.append(result)
            logger.info(f"{test_name} completed in {duration:.2f}s - Status: {result.status.value}")

    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("=" * 80)
        logger.info("Test Execution Summary")
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
                logger.info(
                    f"  {status_symbol} {result.test_type}: "
                    f"{result.passed} passed, {result.failed} failed, "
                    f"{result.skipped} skipped ({result.duration:.2f}s)"
                )

        # Overall summary
        logger.info("\n" + "=" * 80)
        logger.info("Overall Summary:")
        logger.info(f"  Total Tests Passed: {total_passed}")
        logger.info(f"  Total Tests Failed: {total_failed}")
        logger.info(f"  Total Tests Skipped: {total_skipped}")
        logger.info(f"  Total Errors: {total_errors}")
        logger.info(f"  Total Duration: {total_duration:.2f}s")
        logger.info(f"  Success Rate: {(total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0:.2f}%")

        # Save detailed report
        report_file = LOG_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                "success_rate": (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"\nDetailed report saved to: {report_file}")
        logger.info(f"Log file: {log_file}")

        # Print failed tests details
        failed_tests = [r for r in self.results if r.status == TestStatus.FAILED]
        if failed_tests:
            logger.error("\n" + "=" * 80)
            logger.error("Failed Tests Details:")
            for result in failed_tests:
                logger.error(f"\n{result.module} - {result.test_type}:")
                if result.error_details:
                    logger.error(result.error_details[:500] + "..." if len(result.error_details) > 500 else result.error_details)

        return total_failed == 0 and total_errors == 0

    def run_all_tests(self, parallel: bool = False):
        """Run all tests"""
        self.start_time = datetime.now()
        logger.info(f"Starting test execution at {self.start_time}")
        logger.info(f"Running in {'parallel' if parallel else 'sequential'} mode")

        if parallel:
            # Run test suites in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.run_backend_tests),
                    executor.submit(self.run_frontend_tests),
                    executor.submit(self.run_ml_tests),
                    executor.submit(self.run_infrastructure_tests)
                ]
                concurrent.futures.wait(futures)
        else:
            # Run test suites sequentially
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

    parser = argparse.ArgumentParser(description="Master test runner for GoldenSignalsAI V2")
    parser.add_argument("--parallel", action="store_true", help="Run test suites in parallel")
    parser.add_argument("--module", choices=["backend", "frontend", "ml", "infrastructure"],
                       help="Run tests for specific module only")
    args = parser.parse_args()

    runner = TestRunner()

    try:
        if args.module:
            logger.info(f"Running tests for module: {args.module}")
            if args.module == "backend":
                runner.run_backend_tests()
            elif args.module == "frontend":
                runner.run_frontend_tests()
            elif args.module == "ml":
                runner.run_ml_tests()
            elif args.module == "infrastructure":
                runner.run_infrastructure_tests()

            runner.end_time = datetime.now()
            success = runner.generate_report()
        else:
            success = runner.run_all_tests(parallel=args.parallel)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.error("\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
