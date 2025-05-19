# GoldenSignalsAI

AI-powered, multi-agent options trading and arbitrage platform with advanced admin panel, real-time monitoring, and robust security.

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
```

---

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
