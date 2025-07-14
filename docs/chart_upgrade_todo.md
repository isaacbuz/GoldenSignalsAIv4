# GoldenEye AI Prophet Chart Upgrade TODO

This document outlines the remaining tasks to complete the state-of-the-art upgrade for the CentralChart component. Tasks are prioritized and categorized.

## 1. Linter and Type Fixes
- Resolve import errors in `src/components/CentralChart/CentralChart.tsx`:
  - Fix 'Cannot find module react-redux' (ensure @types/react-redux is installed and tsconfig includes it).
  - Correct relative paths for components (e.g., confirm if it's '../TradeSearch/TradeSearch' or via index.ts).
  - Remove duplicate 'updateChartState' imports.
- Fix annotation plugin typing:
  - Use correct props like xValue/yValue for points; cast if needed.
  - Structure plugins object to match Chart.js types (import AnnotationPluginOptions).
- Run `npm run lint -- --fix` and manually address any persistent issues.

## 2. Backend Implementation
- Create real API endpoints in Python (e.g., in `src/api/routes.py` or FastAPI app):
  - `/api/fetch-data`: Use yf.download(symbol, interval=timeframe) to return historical data.
  - `/api/analyze`: Chain agents (e.g., technical/momentum_agent.py, options/options_backtesting.py) to compute aiAnalysis JSON (entries, profitZones, etc.).
- Integrate with existing services like data_fetcher.py and signal_prophet_agent.py for accurate trading signals.

## 3. Redux Integration
- Create `src/redux/actions/chartActions.ts` with updateChartState action (e.g., returns { type: 'UPDATE_CHART', payload: { data, aiAnalysis } }).
- Add chart reducer in `src/redux/reducers/` to handle state updates.
- Wire into CentralChart: Use dispatch(updateChartState()) in handleSubmit and WebSocket onmessage.

## 4. Advanced Features
- **Risk Simulator**: Add slider in OptionsPanel to adjust risk levels, dynamically updating chart annotations via state.
- **Multi-Timeframe Overlay**: Fetch and overlay higher timeframes (e.g., 1d on 5m) using Chart.js datasets.
- **Voice Integration**: In ProphetOrb, add button using Web Speech API to read insights aloud.
- **What-If Scenarios**: Button to simulate volatility changes, re-running AI analysis.

## 5. Styling and UX
- Add GoldenEye theme CSS (e.g., golden borders, orb glow) to components.
- Ensure mobile responsiveness (e.g., orb positions absolute but media queries).
- Add tooltips to annotations with detailed rationale.

## 6. Testing
- Implement tests in generated .test.tsx files (e.g., render TradeSearch, simulate submit).
- Add integration tests for CentralChart with mocked API responses.
- Run `npm test` and fix failures.

## 7. Deployment and CI
- Update scripts/deploy.sh to build and deploy frontend/backend.
- Add GitHub Actions for lint/test on PRs.
- Test locally with `npm start` and a sample symbol like AAPL.

## Prioritization
- High: Fix linters to unblock running.
- Medium: Backend APIs for real data/analysis.
- Low: Advanced features and polish.

Track progress here and mark as done! 