# GoldenSignalsAI Startup Guide

## Quick Start

To start all services with a single command:
```bash
./start.sh
```

This will:
1. Check prerequisites (Python, Node.js, virtual environment)
2. Start databases (PostgreSQL, Redis)
3. Start the backend server on http://localhost:8000
4. Start the frontend on http://localhost:3000

## Master Script Commands

### Start Services
```bash
# Start all services (default)
./start.sh

# Start only backend
./start.sh start --services backend

# Start only frontend
./start.sh start --services frontend

# Start in detached mode (background)
./start.sh start --detached
```

### Stop Services
```bash
# Stop all running services
./start.sh stop
```

### Restart Services
```bash
# Restart all services
./start.sh restart
```

### Check Status
```bash
# Check status of all services
./start.sh status
```

### View Logs
```bash
# View backend logs
./start.sh logs backend

# View frontend logs
./start.sh logs frontend
```

### Install Dependencies
```bash
# Install Python and Node.js dependencies
./start.sh install
```

## Service Endpoints

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

## Startup Modes

### Development Mode (Default)
```bash
./start.sh start --mode dev
```
- Uses local Python virtual environment
- Runs services directly
- Hot-reloading enabled

### Docker Mode
```bash
./start.sh start --mode docker
```
- Uses Docker Compose
- Isolated containers
- Production-like environment

## Troubleshooting

### Port Already in Use
The script automatically kills processes on ports 3000 and 8000 before starting.

### Backend Not Starting
1. Check Python virtual environment:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Check logs:
   ```bash
   ./start.sh logs backend
   ```

### Frontend Not Starting
1. Check Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Check logs:
   ```bash
   ./start.sh logs frontend
   ```

### Database Connection Issues
1. Ensure PostgreSQL is running:
   ```bash
   pg_isready
   ```

2. Ensure Redis is running:
   ```bash
   redis-cli ping
   ```

## Environment Requirements

- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- Redis 6+

## Old Scripts (Removed)

The following scripts have been consolidated into `start.sh`:
- `restart-frontend.sh`
- `run_complete_system.sh`
- `run_evaluation.sh`
- `run_golden_signals.sh`
- `run_phase2.sh`
- `run_phase3.sh`
- `start_all.sh`
- `start_backend_local.sh`
- `start_backend.sh`
- `start_dev.sh`
- `start_frontend_local.sh`
- `start_frontend.sh`
- `start_goldensignals_v3.sh`
- `start_hybrid_system.sh`
- `start-ui.sh`

## Advanced Usage

### Running with Custom Environment
```bash
# Set environment variables
export DATABASE_URL=postgresql://user:pass@localhost/db
export REDIS_URL=redis://localhost:6379

# Start services
./start.sh
```

### Background Services with Logs
```bash
# Start in background
./start.sh start --detached

# Monitor logs in separate terminals
./start.sh logs backend
./start.sh logs frontend
```

### Development Workflow
```bash
# 1. Check status
./start.sh status

# 2. Install/update dependencies
./start.sh install

# 3. Start services
./start.sh

# 4. Make changes (hot-reload will apply them)

# 5. Restart if needed
./start.sh restart

# 6. Stop when done
./start.sh stop
```

## Help

For all available options:
```bash
./start.sh help
``` 