# Claude Instructions for GoldenSignalsAIv4

## Project Overview
This is a FinTech application for analyzing trading signals and options data. The project consists of:
- Python backend (FastAPI) in `src/`
- React TypeScript frontend in `frontend/`
- WebSocket integration for real-time data

## Development Commands

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend server
python src/main.py
```

### Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint

# Run type checking
npm run typecheck
```

## Project Structure
- `src/` - Python backend code
  - `main.py` - FastAPI application entry point
- `frontend/` - React TypeScript frontend
  - `src/components/` - React components
  - `src/services/` - API and WebSocket services
  - `src/utils/` - Utility functions
- `docs/` - Documentation
- `scripts/` - Deployment and utility scripts

## Key Components
- UnifiedDashboard - Main dashboard component
- SignalCard - Displays trading signals
- OptionsChainTable - Options chain visualization
- CentralChart - Main chart component
- WebSocket services for real-time data

## Testing
Before committing changes:
1. Run frontend linting: `cd frontend && npm run lint`
2. Run frontend type checking: `cd frontend && npm run typecheck`
3. Test the application locally

## Git Workflow
- Main branch: `main`
- Create feature branches for new work
- Run tests and linting before committing
- Use descriptive commit messages

## Important Notes
- Always check existing code patterns before making changes
- Follow the existing code style and conventions
- Test WebSocket connections when modifying real-time features
- Ensure TypeScript types are properly defined