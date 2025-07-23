# Chart Component Reference

## 🎯 MAIN CHART COMPONENT

### AITradingChart
**Location**: `/src/components/AIChart/AITradingChart.tsx`

This is the **MAIN** and **ONLY** active chart component in the application.

**Used in**:
- `EnhancedTradingDashboard.tsx`

**Features**:
- ✨ Beautiful dark canvas theme with golden accents
- 📊 Volume display at bottom (no background)
- 🤖 AI prediction trend lines with confidence bounds
- 📈 Technical indicators (SMA, EMA, Bollinger Bands)
- 🌟 Pattern detection with glow effects
- 🔴🟢 Entry/exit arrows with subtle glow
- 🔄 Real-time data updates (no mock data)
- 🎨 Multiple chart types (Candlestick, Mountain, Line, Bar)
- ⚡ WebSocket ready for live data
- 🚫 No simulated data - only real market data

## 📦 Archived Charts
All other chart components have been moved to `/_archived_charts/` to avoid confusion.

## 💡 Important Notes
- The AITradingChart uses **only real data** from the backend
- No mock data or simulated price movements
- Shows error state with retry button when backend is unavailable
- Requires backend server running at http://localhost:8000
