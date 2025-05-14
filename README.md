# GoldenSignalsAI
AI-driven stock trading system using LSTM, XGBoost, LightGBM, FinBERT, SAC RL. Features real-time data, backtesting, SMS/WhatsApp/X alerts, and a Plotly Dash dashboard.

---

## Architecture Overview
![Architecture Diagram](docs/architecture.png)
*_(Replace with your architecture diagram)_*

- **Backend:** FastAPI (REST API), Python 3.10+
- **Frontend:** React & Dash (dashboard)
- **ML:** LSTM, XGBoost, LightGBM, RL, FinBERT
- **Orchestration:** Prefect, Docker, K8s
- **CI/CD:** Poetry, pytest, GitHub Actions, Heroku

## API Usage Example
```bash
curl -X GET http://localhost:8000/signals?symbol=AAPL
```

## Agent & Strategy Explanations
- **Predictive Agents:** Generate signals using ML models
- **Risk Agents:** Manage exposure and risk
- **Sentiment Agents:** Analyze news/social data
- **Strategy Orchestrators:** Combine signals for trading decisions

---

## Security & Secrets
- Never commit real API keys or secrets. Use `.env` files and environment variables. See `.env.example`.

## Frontend & UX
- Dash/React dashboards for real-time monitoring
- Add authentication for production deployments

## CI/CD & Automation
- Automated tests, linting, and deployment via GitHub Actions (see `.github/workflows/ci.yml`).
- Use semantic versioning. Tag releases for changelog and deployment.

---

## Project Structure
- **Presentation**: `api/` (REST API), `frontend/` (Dash dashboard)
- **Application**: `ai_service/`, `events/`, `services/`, `strategies/`, `workflows/`, `monitoring/`
- **Domain**: `trading/`, `models/`, `analytics/`, `portfolio/`
- **Infrastructure**: `data/`, `external_services/`, `event_sourcing/`, `ml_pipeline/`, `monitoring/`, `kyc/`

## Setup
1. Install Poetry: `pip install poetry`
2. Activate env: `conda activate goldensignalsai-py310`
3. Install deps: `poetry install`
4. Run Prefect: `prefect server start`, deploy `daily_cycle.py`
5. Start API: `uvicorn GoldenSignalsAI.presentation.api.main:app --host 0.0.0.0 --port 8000 --reload`
6. Launch dashboard: `python GoldenSignalsAI/presentation/frontend/app/dashboard.py`
7. Run tests: `pytest GoldenSignalsAI/presentation/tests/`

## Download
- [v1.1.0](https://github.com/isaacbuz/GoldenSignalsAI/releases/tag/v1.1.0)
