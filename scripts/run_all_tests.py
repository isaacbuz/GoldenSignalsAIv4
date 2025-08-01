#!/usr/bin/env python3
"""
Comprehensive test runner for GoldenSignalsAI V2
Runs tests in phases and provides detailed reporting.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run_command(self, cmd, phase_name):
        """Run a test command and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {phase_name}...")
        print(f"{'='*60}")

        start = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        duration = time.time() - start

        self.results[phase_name] = {
            'command': cmd,
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

        # Parse pytest output for test counts
        if 'passed' in result.stdout:
            # Extract test counts from pytest output
            import re
            match = re.search(r'(\d+) passed', result.stdout)
            if match:
                self.passed_tests += int(match.group(1))
            match = re.search(r'(\d+) failed', result.stdout)
            if match:
                self.failed_tests += int(match.group(1))

        return result.returncode == 0

    def run_all_tests(self):
        """Run all test phases."""
        self.start_time = datetime.now()

        # Phase 1: Core Agent Tests
        self.run_command(
            "python -m pytest tests/agents/test_rsi_agent.py tests/agents/test_macd_agent.py "
            "tests/agents/test_sentiment_agent.py tests/agents/test_orchestrator.py "
            "tests/agents/test_base_agent.py -v --tb=short",
            "Core Agent Tests"
        )

        # Phase 2: Unit Tests
        self.run_command(
            "python -m pytest tests/unit/ -v --tb=short -x",
            "Unit Tests"
        )

        # Phase 3: Integration Tests
        self.run_command(
            "python -m pytest tests/integration/ -v --tb=short -x",
            "Integration Tests"
        )

        # Phase 4: API Tests
        self.run_command(
            "python -m pytest tests/test_api*.py -v --tb=short -x",
            "API Tests"
        )

        # Phase 5: All Remaining Tests
        self.run_command(
            "python -m pytest tests/ -v --tb=short -x",
            "All Tests"
        )

        # Phase 6: Coverage Report
        self.run_command(
            "python -m pytest tests/ --cov=. --cov-report=html --cov-report=term",
            "Coverage Report"
        )

    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Start Time: {self.start_time}")
        print(f"End Time: {datetime.now()}")
        print(f"Total Duration: {datetime.now() - self.start_time}")
        print(f"\nTotal Tests Passed: {self.passed_tests}")
        print(f"Total Tests Failed: {self.failed_tests}")

        print("\n" + "-"*80)
        print("PHASE RESULTS:")
        print("-"*80)

        for phase, result in self.results.items():
            status = "✅ PASSED" if result['returncode'] == 0 else "❌ FAILED"
            print(f"\n{phase}: {status}")
            print(f"  Duration: {result['duration']:.2f}s")

            if result['returncode'] != 0:
                print(f"  Error Output:")
                print("  " + "\n  ".join(result['stderr'].split('\n')[:10]))

        print("\n" + "="*80)

        # Generate HTML report
        self.generate_html_report()

    def generate_html_report(self):
        """Generate an HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GoldenSignalsAI Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .phase {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
        pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GoldenSignalsAI Test Report</h1>
        <p>Generated: {datetime.now()}</p>
        <p>Total Tests Passed: <span class="passed">{self.passed_tests}</span></p>
        <p>Total Tests Failed: <span class="failed">{self.failed_tests}</span></p>
    </div>

    <h2>Test Phases</h2>
"""

        for phase, result in self.results.items():
            status_class = "passed" if result['returncode'] == 0 else "failed"
            html_content += f"""
    <div class="phase">
        <h3>{phase} - <span class="{status_class}">{'PASSED' if result['returncode'] == 0 else 'FAILED'}</span></h3>
        <p>Duration: {result['duration']:.2f}s</p>
        <p>Command: <code>{result['command']}</code></p>
        <details>
            <summary>Output</summary>
            <pre>{result['stdout'][:5000]}</pre>
        </details>
    </div>
"""

        html_content += """
</body>
</html>
"""

        report_path = Path("test_report.html")
        report_path.write_text(html_content)
        print(f"\nHTML report generated: {report_path.absolute()}")


def main():
    """Main entry point."""
    print("🚀 GoldenSignalsAI Comprehensive Test Runner")
    print("=" * 80)

    runner = TestRunner()

    try:
        runner.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {e}")
    finally:
        runner.print_summary()

    # Exit with appropriate code
    sys.exit(0 if runner.failed_tests == 0 else 1)


if __name__ == "__main__":
    main()
