#!/usr/bin/env python3
"""
GoldenSignalsAI V3 - Complete Setup & Management Script
"""
import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path

class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a bold header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

def run_command(cmd, description, cwd=None, check=True):
    """Run a command with nice output"""
    print_info(f"{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        if result.returncode == 0:
            print_success(f"{description} completed!")
            return True, result.stdout
        else:
            print_error(f"{description} failed!")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print_error(f"{description} failed with exception: {e}")
        return False, str(e)

class GoldenSignalsSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.frontend_dir = self.project_root / "frontend"
        self.backend_running = False
        self.frontend_running = False
        
    def check_prerequisites(self):
        """Check if all required tools are installed"""
        print_header("Checking Prerequisites")
        
        prerequisites = {
            "Python 3.11+": "python3 --version",
            "Node.js 16+": "node --version",
            "npm": "npm --version",
            "Git": "git --version"
        }
        
        all_good = True
        for tool, cmd in prerequisites.items():
            success, output = run_command(cmd, f"Checking {tool}", check=False)
            if not success:
                print_error(f"{tool} is not installed!")
                all_good = False
            else:
                print_success(f"{tool}: {output.strip()}")
                
        return all_good
    
    def setup_python_env(self):
        """Set up Python virtual environment and dependencies"""
        print_header("Setting up Python Environment")
        
        # Create virtual environment if it doesn't exist
        venv_path = self.project_root / ".venv"
        if not venv_path.exists():
            run_command("python3 -m venv .venv", "Creating virtual environment")
        else:
            print_info("Virtual environment already exists")
        
        # Activate and install dependencies
        pip_cmd = f"{venv_path}/bin/pip"
        python_cmd = f"{venv_path}/bin/python"
        
        run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
        
        # Install additional missing dependencies
        missing_deps = [
            "slowapi==0.1.9",
            "itsdangerous",
            "PyJWT",
            "python-multipart"
        ]
        
        for dep in missing_deps:
            run_command(f"{pip_cmd} install {dep}", f"Installing {dep}")
        
        # Install main requirements
        success, _ = run_command(
            f"{pip_cmd} install -r requirements.txt", 
            "Installing Python dependencies"
        )
        
        return success
    
    def create_ml_models(self):
        """Create ML models if they don't exist"""
        print_header("Setting up ML Models")
        
        model_script = self.project_root / "ml_training" / "create_basic_models.py"
        if model_script.exists():
            python_cmd = f"{self.project_root}/.venv/bin/python"
            success, _ = run_command(
                f"{python_cmd} {model_script}",
                "Creating ML models",
                cwd=str(self.project_root / "ml_training")
            )
            return success
        else:
            print_warning("ML model creation script not found")
            return False
    
    def setup_database(self):
        """Initialize database"""
        print_header("Setting up Database")
        
        # Create SQLite database directory
        db_dir = self.project_root / "data"
        db_dir.mkdir(exist_ok=True)
        
        print_success("Database directory created")
        print_info("Using SQLite for development (PostgreSQL optional for production)")
        
        return True
    
    def setup_frontend(self):
        """Set up frontend dependencies"""
        print_header("Setting up Frontend")
        
        if not self.frontend_dir.exists():
            print_error("Frontend directory not found!")
            return False
        
        # Clean install
        node_modules = self.frontend_dir / "node_modules"
        if node_modules.exists():
            print_info("Cleaning existing node_modules...")
            shutil.rmtree(node_modules)
        
        # Install dependencies
        success, _ = run_command(
            "npm install",
            "Installing frontend dependencies",
            cwd=str(self.frontend_dir)
        )
        
        return success
    
    def test_backend(self):
        """Test if backend can start"""
        print_header("Testing Backend")
        
        python_cmd = f"{self.project_root}/.venv/bin/python"
        
        # Start backend in background
        print_info("Starting backend server...")
        backend_process = subprocess.Popen(
            [python_cmd, "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for startup
        time.sleep(5)
        
        # Test health endpoint
        success, output = run_command(
            "curl -s http://localhost:8000/health || echo 'Failed'",
            "Testing health endpoint",
            check=False
        )
        
        # Kill the process
        backend_process.terminate()
        backend_process.wait()
        
        if "healthy" in output.lower():
            print_success("Backend is working!")
            return True
        else:
            print_error("Backend test failed")
            return False
    
    def create_env_file(self):
        """Create .env file with defaults"""
        print_header("Creating Environment Configuration")
        
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = """# GoldenSignalsAI Environment Configuration
DEBUG=True
SECRET_KEY=your-secret-key-here-change-in-production
DATABASE_URL=sqlite:///./data/goldensignals.db
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional: API Keys (add your own)
# ALPHA_VANTAGE_API_KEY=your-key-here
# IEX_CLOUD_API_KEY=your-key-here
# OPENAI_API_KEY=your-key-here
"""
            env_file.write_text(env_content)
            print_success(".env file created with default values")
            print_warning("Remember to update SECRET_KEY and JWT_SECRET_KEY for production!")
        else:
            print_info(".env file already exists")
        
        return True
    
    def create_startup_scripts(self):
        """Create convenient startup scripts"""
        print_header("Creating Startup Scripts")
        
        # Backend startup script
        backend_script = self.project_root / "start_backend.sh"
        backend_content = """#!/bin/bash
source .venv/bin/activate
echo "🚀 Starting GoldenSignalsAI Backend..."
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""
        backend_script.write_text(backend_content)
        backend_script.chmod(0o755)
        
        # Frontend startup script
        frontend_script = self.project_root / "start_frontend.sh"
        frontend_content = """#!/bin/bash
cd frontend
echo "🚀 Starting GoldenSignalsAI Frontend..."
npm run dev
"""
        frontend_script.write_text(frontend_content)
        frontend_script.chmod(0o755)
        
        # Combined startup script
        all_script = self.project_root / "start_all.sh"
        all_content = """#!/bin/bash
echo "🚀 Starting GoldenSignalsAI Full Stack..."

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend
./start_frontend.sh &
FRONTEND_PID=$!

echo "✅ Backend PID: $BACKEND_PID"
echo "✅ Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
"""
        all_script.write_text(all_content)
        all_script.chmod(0o755)
        
        print_success("Startup scripts created!")
        return True
    
    def run_full_setup(self):
        """Run the complete setup process"""
        print_header("🚀 GoldenSignalsAI V3 Complete Setup")
        
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Environment File", self.create_env_file),
            ("Python Environment", self.setup_python_env),
            ("ML Models", self.create_ml_models),
            ("Database", self.setup_database),
            ("Frontend", self.setup_frontend),
            ("Backend Test", self.test_backend),
            ("Startup Scripts", self.create_startup_scripts)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    print_warning(f"{step_name} had issues but continuing...")
            except Exception as e:
                print_error(f"{step_name} failed with error: {e}")
                failed_steps.append(step_name)
        
        print_header("Setup Complete!")
        
        if failed_steps:
            print_warning(f"Some steps had issues: {', '.join(failed_steps)}")
            print_info("You may need to address these manually")
        else:
            print_success("All steps completed successfully!")
        
        print("\n📚 Quick Start Guide:")
        print("1. Start backend only:   ./start_backend.sh")
        print("2. Start frontend only:  ./start_frontend.sh")
        print("3. Start full stack:     ./start_all.sh")
        print("\n🌐 Access Points:")
        print("   Frontend:  http://localhost:3000")
        print("   Backend:   http://localhost:8000")
        print("   API Docs:  http://localhost:8000/docs")
        
        return len(failed_steps) == 0

def main():
    """Main entry point"""
    setup = GoldenSignalsSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "backend":
            setup.test_backend()
        elif command == "frontend":
            setup.setup_frontend()
        elif command == "models":
            setup.create_ml_models()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python setup_goldensignals.py [backend|frontend|models]")
    else:
        # Run full setup
        setup.run_full_setup()

if __name__ == "__main__":
    main() 