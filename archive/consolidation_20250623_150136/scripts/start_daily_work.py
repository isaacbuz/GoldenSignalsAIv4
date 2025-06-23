#!/usr/bin/env python3
"""
GoldenSignalsAI V2 - Daily Work Startup Script
This script helps start your daily work on the execution plan.
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def run_command(cmd, description, check=True):
    """Run a command and display the result."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 or not check:
            print(f"✅ {description} - Complete")
            return result.stdout
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return None

def check_environment():
    """Check if the environment is properly set up."""
    print_header("Environment Check")
    
    # Check if we're in the right directory
    cwd = os.getcwd()
    if "GoldenSignalsAI_V2" not in cwd:
        print("❌ Not in GoldenSignalsAI_V2 directory!")
        print(f"   Current: {cwd}")
        return False
    
    # Check if virtual environment is activated
    if not sys.prefix.endswith('.venv'):
        print("⚠️  Virtual environment not activated")
        print("   Run: source .venv/bin/activate")
        return False
    
    print("✅ Environment ready")
    return True

def run_quality_tests():
    """Run the quality test suite."""
    print_header("Running Quality Tests")
    output = run_command("python run_quality_tests.py", "Quality Tests")
    
    if output and "Success Rate: 100.0%" in output:
        print("🎉 All quality tests passing!")
    elif output:
        print("⚠️  Some quality tests may be failing")
        print("   Run 'python run_quality_tests.py' for details")

def check_api_status():
    """Check if the backend API is running."""
    print_header("API Status Check")
    
    # Check if backend is running
    result = run_command("curl -s http://localhost:8000/health", "Backend health check", check=False)
    
    if result and "healthy" in result.lower():
        print("✅ Backend API is running")
    else:
        print("⚠️  Backend API is not running")
        print("   To start: python standalone_backend_optimized.py &")

def show_current_phase():
    """Display the current phase and tasks."""
    print_header("Current Phase Status")
    
    # Read the execution tracker if it exists
    tracker_path = Path("EXECUTION_TRACKER.md")
    if tracker_path.exists():
        with open(tracker_path, 'r') as f:
            content = f.read()
            
        # Extract current phase
        for line in content.split('\n'):
            if "Current Phase" in line:
                print(f"📍 {line.strip()}")
                break
        
        # Show TODO tasks
        print("\n📋 Today's TODO items:")
        todo_count = 0
        for line in content.split('\n'):
            if "⬜ TODO" in line and "|" in line:
                task = line.split("|")[0].strip()
                if task:
                    print(f"   - {task}")
                    todo_count += 1
                if todo_count >= 5:  # Show max 5 TODOs
                    print("   ... and more")
                    break

def show_recent_errors():
    """Check for recent errors in logs."""
    print_header("Recent Errors")
    
    log_files = ["backend.log", "test_logs/latest.log"]
    errors_found = False
    
    for log_file in log_files:
        if Path(log_file).exists():
            result = run_command(f"grep -i error {log_file} | tail -5", f"Checking {log_file}", check=False)
            if result and result.strip():
                print(f"\n❌ Errors in {log_file}:")
                print(result)
                errors_found = True
    
    if not errors_found:
        print("✅ No recent errors found")

def show_quick_commands():
    """Display quick commands for today's work."""
    print_header("Quick Commands for Today")
    
    commands = {
        "Start backend": "python standalone_backend_optimized.py &",
        "Run specific test": "python -m pytest tests/unit/test_data_quality.py -v",
        "Check coverage": "python -m pytest tests/unit --cov=src --cov-report=term-missing",
        "Update tracker": "code EXECUTION_TRACKER.md",
        "View logs": "tail -f backend.log",
        "Test API": "curl http://localhost:8000/api/v1/signals",
    }
    
    for desc, cmd in commands.items():
        print(f"  {desc:20} → {cmd}")

def create_daily_note():
    """Create a daily note file for tracking progress."""
    print_header("Daily Progress Note")
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    note_file = f"daily_notes/{today}.md"
    
    os.makedirs("daily_notes", exist_ok=True)
    
    if not Path(note_file).exists():
        with open(note_file, 'w') as f:
            f.write(f"# Daily Progress - {today}\n\n")
            f.write("## Tasks Completed\n- \n\n")
            f.write("## Challenges\n- \n\n")
            f.write("## Tomorrow's Goals\n- \n\n")
            f.write("## Notes\n- \n")
        
        print(f"📝 Created daily note: {note_file}")
    else:
        print(f"📝 Today's note exists: {note_file}")
    
    print(f"   Edit: code {note_file}")

def main():
    """Main execution flow."""
    print("\n🚀 GoldenSignalsAI V2 - Daily Startup")
    print(f"📅 {datetime.datetime.now().strftime('%A, %B %d, %Y')}")
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues first!")
        return 1
    
    # Run checks
    run_quality_tests()
    check_api_status()
    show_current_phase()
    show_recent_errors()
    show_quick_commands()
    create_daily_note()
    
    # Final message
    print_header("Ready to Start!")
    print("💪 Good luck with today's tasks!")
    print("📖 Don't forget to update EXECUTION_TRACKER.md")
    print("🔍 Run 'python run_quality_tests.py' after major changes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 