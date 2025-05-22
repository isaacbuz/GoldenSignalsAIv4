# Running the Vite + React Frontend

To run the Vite + React frontend, navigate to the project directory and run the following command:

```
npm run dev
```

This will start the development server and make the application available at `http://localhost:3000`.

## Admin Dashboard Features (2025)

- **🔒 Secure API Layer:** All admin API calls use a JWT/session token via the `authFetch` wrapper. 401 responses automatically redirect to `/login`.
  - On login, store your token with: `localStorage.setItem("token", "<your_token>")`.
- **📄 Export to CSV/JSON:** Download any table (signals, agent stats, audit log) as CSV or JSON. Click the export buttons above each table.
- **📡 Agent Health Monitor:** Live dashboard for agent status (online/offline), uptime, ping, and error count. Auto-refreshes every 30 seconds.
- **📅 Scheduled Reports:** Schedule daily/weekly reports (signals, agent stats, audit log) as CSV/JSON. Managed via the admin UI, all actions are audit-logged.
- **📝 Audit Logging:** All admin actions (agent control, report scheduling, etc.) are tracked in a live audit log, visible and exportable.

## Configuring the Backend API URL

To configure the backend API URL, create a new file named `.env` in the project root directory and add the following line:

```
VITE_API_URL=https://your-backend-api-url.com
```

Replace `https://your-backend-api-url.com` with the actual URL of your backend API.

Note: Make sure to restart the development server after making changes to the `.env` file.
