<<<<<<< HEAD

# GoldenSignalsAI

[![Build Status](https://github.com/isaacbuz/GoldenSignalsAI/actions/workflows/test.yml/badge.svg)](https://github.com/isaacbuz/GoldenSignalsAI/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Project Goal
GoldenSignalsAI is an AI-powered stock prediction and trade suggestion platform. It predicts whether a stock will go **up or down** in a given timeframe and suggests actionable entry and exit points for **calls or puts**. The platform leverages advanced machine learning, real-time and historical market data from multiple sources (including Financial Modeling Prep, Alpha Vantage, Polygon, Finnhub, and more), and a modern dashboard UI to empower traders with intelligent, actionable insights.

---

## Features

- 📈 **Stock Direction Prediction**: Predicts up/down movement for any supported stock and timeframe using AI/ML models.
- 🎯 **Trade Suggestions**: Recommends entry and exit points for calls or puts based on predictions and market context.
- 🤖 **Multi-Agent, Multi-Source Data**: Integrates data from Financial Modeling Prep, Alpha Vantage, Polygon, Finnhub, and more for robust, reliable signals.
- 📊 **Performance & Signal Dashboard**: Modern, responsive dashboard for viewing predictions, trade suggestions, and performance metrics.
- 🔒 **Secure Admin API Layer**: All admin API requests use a JWT/session token, injected via a robust `authFetch` wrapper. 401 responses trigger auto-redirect to `/login`.
- 📈 **Export to CSV/JSON**: Download any table (signals, agent stats, audit log) as Excel-ready CSV or JSON for audits, ML, or archiving.
- 🤖 **Agent Health Monitor**: Live dashboard for agent online/offline status, uptime, ping, and error count. Auto-refreshes every 30s.
- 📅 **Scheduled Reports System**: Schedule daily/weekly reports (signals, agent stats, audit log) as CSV/JSON. Fully managed via the admin UI, with all actions audit-logged.
- 📝 **Audit Logging**: All critical admin actions (agent control, report scheduling, etc.) are tracked in a live audit log, visible in the dashboard and exportable.
- ⚡ **Dynamic Ticker Search & Autocomplete**: Instantly search and select tickers with live validation and autocomplete.
- 🔔 **Visual Alerts & Monitoring**: Alerts for prediction confidence, model health, and system status.
- 🛡️ **Role-Based Admin Panel**: User management, access control, and audit logging.
- 🔒 **Secure, Extensible Architecture**: Modular backend and frontend, with best practices for secrets and user authentication.

---

## Frontend Architecture & Features

- **Modern UI/UX**: Built with React, Tailwind, shadcn/ui, and robust accessibility.
- **Global Shell**: Planned persistent sidebar, header, and dynamic content area for all pages.
- **Pages:**
  - Dashboard: Real-time signals, equity curve, summary KPIs, alerts, heatmap, recent activity.
  - Signals: Interactive price chart with buy/sell/hold markers, signal history, confidence bars.
  - Watchlist: Ticker cards, sparklines, P/L, drag-and-drop reorder.
  - Backtest: Equity curve, drawdown, performance metrics, correlation matrix.
  - Agents: Toggle agent list, configuration panel.
  - Settings: API keys, notification preferences, profile.
- **Advanced Features:**
  - Real-time WebSocket feeds for ticks and signals.
  - Interactive chart overlays, drawing tools, multi-timeframe toggles.
  - Guided tour, quick-action toolbar, session replay, notifications.
  - Unified error handling: UI never crashes on missing env or API errors—shows warnings instead.
- **Environment Sync:**
  - Use `scripts/sync-env.sh` to keep frontend and backend envs in sync.
  - All frontend env vars must start with `VITE_`.

---

## Admin Dashboard: Power Features

### Secure API Layer
- All admin API calls use `authFetch`, which injects the JWT/session token from localStorage and auto-redirects to `/login` on 401.
- **Usage:** On login, store your token via `localStorage.setItem("token", "<your_token>")`.

### Export to CSV/JSON
- Download any table (signals, agent stats, audit log) as CSV or JSON with one click.
- **Usage:** Click the Export CSV/JSON buttons above each table in the admin dashboard.

### Agent Health Monitor
- View all agent statuses (online/offline), uptime, ping, and error count in real time.
- **Usage:** See the "Agent Health Monitor" panel in the admin dashboard. Data auto-refreshes every 30 seconds.

### Scheduled Reports System
- Schedule daily or weekly exports of signals, agent stats, or audit logs in CSV or JSON format.
- **Usage:** Use the "Schedule Automated Reports" panel in the admin dashboard. All scheduling actions are tracked in the audit log.

### Audit Logging
- All admin actions (agent control, report scheduling, etc.) are tracked and visible in the live audit log panel.
- **Usage:** See the "Admin Audit Log" section. Export logs for compliance or review.

---
- **Conda Environment:**
  - Preferred environment: `goldensignalsai` (see memory notes).

---

## Screenshots & Demo

![Dashboard Screenshot](docs/dashboard_placeholder.png)
*Replace this with a real dashboard screenshot!*

**Demo GIF:**
![Demo GIF](docs/demo_placeholder.gif)
*Replace this with a short GIF/video of the dashboard in action (e.g., use Recordit or Loom).*

---

## Quickstart

### 1. Clone & Install

#### Using Poetry (Recommended)
```bash
git clone https://github.com/isaacbuz/GoldenSignalsAI.git
cd GoldenSignalsAI
pip install poetry
poetry install
```

### 2. Environment Setup (Unified)
- Use a single `.env` file at the project root for all secrets, API keys, and environment variables.
- All frontend variables must be prefixed with `VITE_` (e.g. `VITE_API_URL`, `VITE_FMP_API_KEY`).
- Backend variables are unprefixed and used by FastAPI and agents.
- To sync frontend variables, run:
  ```bash
  bash scripts/sync-env.sh
  ```
- This will update `presentation/frontend/.env` with only the `VITE_` variables from the root `.env`.
- **Never commit real secrets!** Use environment variables or vaults for production.
- Create and activate the conda environment (recommended):
  ```bash
  conda create -n goldensignalsai python=3.10
  conda activate goldensignalsai
  pip install poetry
  poetry install
  ```

### 3. Logging Configuration
- Logging is now centralized in `logging.yaml` for both console and file output. You can adjust log levels, formats, and destinations there.

### 4. Pre-commit Hooks
- Install pre-commit hooks to enforce code style and quality:
  ```bash
  pre-commit install
  ```
- This will run Black, isort, and flake8 on every commit.

### 5. Docker & Infra Structure
- All Dockerfiles are now grouped under `/docker`.
- Infra scripts and deploy scripts are grouped under `/scripts`.
- Update your `docker-compose.yml` and CI/CD scripts to reference these new paths.

### 6. Run the Platform
```bash
# Start FastAPI backend (in the goldensignalsai conda env)
poetry run python main.py

# Sync frontend .env (in project root)
bash scripts/sync-env.sh

# Start React frontend (from presentation/frontend)
npm install
npm run dev
```

---

## Docker & Compose

To run the full stack with Docker Compose:
```bash
docker-compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Prometheus & Grafana Monitoring

- Prometheus config: `prometheus/prometheus.yml`
- Start via Docker Compose or run Prometheus manually
- Add Grafana dashboards for real-time monitoring

---

## GitHub Actions CI

Automated tests run on every push via [GitHub Actions](.github/workflows/test.yml).

---

## Testing

Run backend tests with pytest:
```bash
poetry run pytest
```
Or using pip:
```bash
pytest
```
Sample test: `tests/test_health.py`

---

## Error Handling & Robustness
- Backend endpoints use robust exception handling and logging
- Frontend dashboard features loading indicators, error messages, and retry logic for API/WebSocket failures
- FastAPI auto-generates OpenAPI/Swagger docs at `/docs`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Architecture Overview
![Architecture Diagram](docs/architecture.png)

- **Backend:** FastAPI (REST API, trading logic, admin endpoints)
- **Frontend:** React (dashboard UI, admin panel), Dash (advanced analytics)
- **Agents:** Modular Python classes for each data source (Alpha Vantage, Finnhub, Polygon, Benzinga, Bloomberg, StockTwits)
- **Arbitrage:** AI-powered arbitrage agents and executor, real-time monitoring
- **Deployment:** Poetry, Docker Compose, .env for secrets
- **Security:** Firebase multi-auth, RBAC, audit logging, no hardcoded keys

---

## Features

### ⚡️ Trading & Arbitrage
- Real-time options trading signals, arbitrage detection, and execution
- Modular, extensible agent framework (add new data sources easily)
- Handles API key rotation, missing/expired keys gracefully

### 📊 Admin Panel (React)
- **Multi-provider authentication:** Firebase (email/password, Google, GitHub)
- **Role-based access control:** Admin/operator/user via Firebase custom claims
- **Live monitoring:**
  - Performance charts (CPU, memory, uptime)
  - Agent health/heartbeat, error rates
  - Queue/task status
  - Real-time logs
- **Agent controls:** Restart, disable agents (admin only)
- **Dynamic Ticker Search:** Users can enter any stock ticker in the dashboard, validated live against backend data sources (no more hardcoded dropdowns).
- **User management:**
  - List users, change roles, enable/disable, invite, reset password, delete
  - All sensitive actions are audit-logged
- **Alerts:** Visual alerts for unhealthy agents, high queue depth, errors
- **Modular Admin Panel:** Admin panel is modularized into subcomponents for agents, users, queue, analytics, and alerts for maintainability and extensibility.
- **Onboarding:** Built-in onboarding modal for new admins
- **Accessibility:** Keyboard navigation, ARIA labels, color contrast

### 🛡️ Security & Compliance
- All secrets in `.env`, never committed
- Firebase ID token required for all admin endpoints
- Audit logging for all sensitive admin/user actions
- RBAC enforced on backend and frontend

---

## Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/isaacbuz/GoldenSignalsAI.git
cd GoldenSignalsAI
pip install poetry
poetry install
```

### 2. Environment Setup
- Copy `.env.example` to `.env` and fill in:
  - All API keys (Alpha Vantage, Finnhub, etc.)
  - Firebase Admin SDK path
  - `FIREBASE_WEB_API_KEY` (from Firebase Console)
  - Any other required secrets
- Create and activate the conda environment:
  ```bash
  conda create -n goldensignalsai python=3.10
  conda activate goldensignalsai
  pip install poetry
  poetry install
  ```

### 3. Run the Platform
```bash
# Start FastAPI backend
uvicorn GoldenSignalsAI.presentation.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start React frontend (from presentation/frontend)
npm install
npm start

# The frontend and backend are managed in a monorepo structure with centralized config (`config/` and `.env`).

# (Optional) Start Dash analytics dashboard
python GoldenSignalsAI/presentation/frontend/app/dashboard.py
=======
# 🧠 GoldenSignalsAI

GoldenSignalsAI is a modular, multi-agent AI signal platform for generating and evaluating real-time trading signals using data from TradingView, social sentiment (X, Reddit, LinkedIn), and machine learning strategies.

---

## 📦 Project Structure

backend/
├── agents/
│ ├── base.py # Abstract base class for all signal agents
│ ├── agent_manager.py # Runs all agents and aggregates/votes on their signals
│ ├── signal_engine.py # Top-level orchestrator (entrypoint for signal engine)
│ ├── ml/ # ML-driven agents and sentiment aggregator
│ ├── tv/ # TradingView scraping and signal agents
│ └── nlp/ # Text-based sentiment agents
│
├── api/
│ └── routes/
│ ├── sentiment.py # Crowd sentiment endpoint
│
├── db/
│ └── signal_logger.py # Persists signals in PostgreSQL
│
├── core/
│ └── websocket.py # WebSocket broadcasting to frontend
│
├── nlp/
│ └── sentiment_engine.py # Core NLP sentiment scoring engine
│
frontend/
├── components/
│ └── signalPanel/
│ ├── SignalCard.tsx # Realtime signal UI card
│ └── TrendingSentimentBar.tsx # Top sentiment recommendations
│
└── pages/
└── index.tsx # Dashboard entry point

---

## 🧠 How It Works

1. Agents (e.g., TV, ML, NLP) are registered and executed by the `AgentManager`.
2. Each agent returns a `signal = {"symbol", "action", "confidence", "agent"}`.
3. AgentManager aggregates signals and votes on consensus.
4. Signals are pushed to frontend via WebSocket and saved in PostgreSQL.
5. Sentiment from multiple platforms influences signal confidence.

---

## 🚀 Quick Start

Run backend server:
```bash
uvicorn main:app --reload
```

Run frontend:
```bash
cd frontend
npm run dev
```

Run agents manually:
```bash
python backend/agents/signal_engine.py --symbol AAPL
```

---

## ✅ Roadmap

- [x] Multi-agent orchestration
- [x] TradingView & ML agents
- [x] Social sentiment enhancement
- [x] Real-time signal pushing
- [ ] Agent performance dashboard
- [ ] CLI simulation tool
- [ ] Automated backtester

---

## 🧪 Test Runner (coming soon)

Simulate historical signals:
```bash
python test_agent_runner.py --symbol AAPL --date 2024-05-01
>>>>>>> a9235431 (Initial commit: Add GoldenSignalsAI_Merged_Final with ML agents, retraining automation, and advanced frontend visualization)
```

---

<<<<<<< HEAD
## Admin Panel Usage

- Access the admin panel from the main dashboard UI (localhost:8080)
- Log in with Firebase (email/password, Google, GitHub)
- Admins can:
  - Monitor performance, agent health, queue
  - Manage users and roles
  - Restart/disable agents
  - Invite, reset password, or delete users
- All actions are audit-logged to `./logs/admin_audit.log`
- Alerts are shown for system or agent issues

---

## Security Best Practices
- Never commit real API keys or secrets
- Use `.env` for all secrets, reference with `os.getenv`
- Restrict admin access via Firebase custom claims
- All admin endpoints require valid Firebase ID token
- Regularly review `admin_audit.log` for compliance

---

## Extending & Customizing
- **Add new agents:** Create a new Python agent class and register in the backend
- **Add new admin features:** Add endpoints in FastAPI, then UI in React
- **Integrate new auth providers:** Enable in Firebase console and update frontend
- **Customize onboarding:** Edit `AdminOnboardingModal.js`

---

## Troubleshooting
- **Auth errors:** Check Firebase config and service account
- **API errors:** Check `.env` for missing/expired keys
- **Frontend/backend not connecting:** Confirm CORS and port settings
- **Audit log missing:** Ensure `logs/` directory exists and is writable
- **Ticker validation errors:** If you see 'Invalid ticker symbol', ensure the backend is running and the symbol exists in supported data sources.

---

## Project Structure
- `presentation/api/` — FastAPI backend (admin endpoints, trading logic, ticker validation endpoint)
- `presentation/frontend/` — React frontend (dashboard, admin panel, dynamic ticker search)
- `infrastructure/` — Agent definitions, data sources
- `config/` — Centralized configuration for API URLs and environment variables
- `logs/` — System and audit logs
- `.env` — Secrets and API keys (never commit!)

---

## License & Credits
- MIT License
- Created by Isaac Buz and contributors

---

## Need Help?
- See the onboarding modal in the admin panel
- Open an issue on GitHub
- Contact project maintainers

# GoldenSignalsAI: AI-Powered Options Trading Signal Generator 🚀📈

## Overview

GoldenSignalsAI is an advanced, multi-agent AI system designed to generate intelligent trading signals for options trading. Leveraging machine learning, real-time data processing, and sophisticated risk management strategies.

### Key Features

- 🤖 Multi-Agent Architecture
- 🧠 Machine Learning Signal Generation
- 📊 Real-Time Market Data Processing
- 🛡️ Advanced Risk Management
- 🔍 Sentiment Analysis Integration

## Technical Architecture

### Components
- **Backend**: FastAPI
- **Machine Learning**: Custom AI Models
- **Data Processing**: Streaming & Batch Processing
- **Deployment**: Docker, Kubernetes Support

### System Design Principles
- Modular Microservices
- Dependency Injection
- Comprehensive Error Handling
- Performance Monitoring
- Adaptive Machine Learning

## Getting Started

### Prerequisites
- Python 3.10+
- Poetry
- Docker (optional)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/GoldenSignalsAI.git
cd GoldenSignalsAI
```

2. Install dependencies
```bash
poetry install
```

3. Configure Environment
- Copy `.env.example` to `.env`
- Fill in required API keys

4. Run the application
```bash
poetry run python main.py
```

## Configuration

Customize `config.yaml` for:
- API Credentials
- Feature Flags
- Trading Parameters
- Notification Settings

## Testing

Run comprehensive test suite:
```bash
poetry run pytest
```

## Deployment Options

### Local Development
```bash
poetry run uvicorn main:app --reload
```

### Docker
```bash
docker-compose up --build
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Machine Learning Community
- Open Source Contributors
- Financial Technology Innovators

---

## Robustness, Validation, and Error Handling (Advanced Agents)

All advanced agent endpoints implement:

- **Pydantic request/response models**: Strict input validation for all endpoints.
- **Structured error responses**: All errors are returned as JSON with an `error` field.
- **Logging**: All requests and errors are logged for debugging and monitoring.
- **Edge case handling**: Endpoints handle empty/invalid input and return clear error messages.

### Example Error Response

```json
{
  "average_score": 0.0,
  "raw_results": [{"error": "No texts provided."}]
}
```

### Example Log Output

```
INFO:root:FinBERT /analyze called with 0 texts.
ERROR:root:FinBERT error: No texts provided.
```

### Negative Test Coverage

A dedicated test suite (`tests/test_api_endpoints_negative.py`) ensures endpoints handle invalid input gracefully and never crash. It covers scenarios such as:
- Empty input lists
- Insufficient data for time series models
- Invalid or missing required fields
- Model not trained/loaded

Run all tests (including negative cases):
```bash
pytest tests/
```

For more details, see the `tests/` directory and API endpoint docstrings.
=======
## 📎 License

MIT — open-source and extensible.

## Setup

1. Clone repo
2. Install dependencies (`npm install` in frontend and backend)
3. Create `.env` file

## Running

- Frontend: `cd frontend && npm run dev`
- Backend: `cd backend && npm run dev` (if applicable)

## API Endpoints

- `/api/signal/check`
- `/api/signal/save`
ldenSignalsAI v4

Modular, AI-powered trading signal engine with live data, multiple agents, and full frontend/backend stack.

## ✅ Quickstart

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Start the backend
```
uvicorn main:app --reload
```

### 3. Access endpoints
- `GET /api/analyze?ticker=TSLA`
- `POST /api/train`
- `GET /api/logs`

### 4. Run tests
```
python test_analyze_pipeline.py
```

### 5. Frontend
Integrate your `frontend/` directory with your preferred React + Tailwind build system (Next.js recommended).

---

GoldenSignalsAI is designed for modular AI research, options signals, and sentiment-enhanced technical analysis.
>>>>>>> a9235431 (Initial commit: Add GoldenSignalsAI_Merged_Final with ML agents, retraining automation, and advanced frontend visualization)
