// presentation/frontend/src/App.js
// Purpose: Main entry point for the GoldenSignalsAI frontend. Handles routing, authentication state, and renders the appropriate component (dashboard, arbitrage, admin panel, etc.) based on user role and authentication status. Ensures a clean separation between user and admin experiences.

import React, { useState, useEffect, useRef } from "react";
import './App.css';
import API_URL from './config';
import Arbitrage from './Arbitrage';
import AdminPanel from "./AdminPanel";
import ErrorBoundary from "./ErrorBoundary";
import Box from '@mui/material/Box';
import CssBaseline from '@mui/material/CssBaseline';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';
import Divider from '@mui/material/Divider';
import Avatar from '@mui/material/Avatar';
import Tooltip from '@mui/material/Tooltip';
import SettingsIcon from '@mui/icons-material/Settings';
import { deepPurple } from '@mui/material/colors';
import Login from './Login';
import { createChart } from 'lightweight-charts';

// Add helper for JWT token retrieval if not present
function getJwtToken() {
  return localStorage.getItem('jwt_token');
}
// Recharts imports


// Lightweight Charts
// (imported above)



/**
 * App Component: Handles authentication state, routing, and rendering of components.
 * 
 * @returns {JSX.Element} The rendered App component.
 */
function App() {
  // State for tracking if the user is an admin
  // Bypass auth: set isAdmin and user directly for testing
  const [isAdmin, setIsAdmin] = useState(false); // Set to true to test admin panel
  const [user, setUser] = useState({ displayName: 'Test User', email: 'test@example.com' });
  const [activeTab, setActiveTab] = useState('signals');
  const [symbol, setSymbol] = useState('AAPL');
  const [data, setData] = useState(null);
  const [token, setToken] = useState('dummy-token'); // Bypass auth with dummy token
  // Chart controls
  const [chartTimeframe, setChartTimeframe] = useState('D');
  const [chartType, setChartType] = useState('candlestick');
  // Token/session expiry state
  // Auth bypass: remove  state entirely
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [indicatorToggles, setIndicatorToggles] = useState({ MA_Confluence: true, RSI: false, MACD_Strength: false, VWAP_Score: false, Volume_Spike: false });
  // Drawing tools state
  const [drawingMode, setDrawingMode] = useState(null); // 'trendline' | 'annotation' | null
  const [trendlines, setTrendlines] = useState([]); // [{start: {time, price}, end: {time, price}}]
  const [pendingLine, setPendingLine] = useState(null); // {start: {time, price}}
  const [annotations, setAnnotations] = useState([]); // [{time, price, text}]
  // Markers and indicators
  const [tradeMarkers, setTradeMarkers] = useState([]);
  const [indicators, setIndicators] = useState({});
  // Chart container ref
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const candleSeriesRef = useRef();
  const lineSeriesRef = useRef();
  const indicatorSeriesRef = useRef();
  // Login form state (must be here, not inside conditional)
  // Login state and handlers fully removed for test mode

  // Listen for token expiry (Firebase)
  // Auth bypass: remove useEffect for token expiry

  useEffect(() => {
    setLoading(true);
    setError(null);
    let ws;
    let wsActive = false;
    let reconnectTimer = null;
    function connectWS() {
      // Convert API_URL to WS_URL for WebSocket connection
      let WS_URL = API_URL.replace(/^http/, 'ws');
      ws = new window.WebSocket(`${WS_URL}/ws/ohlcv?symbol=${symbol}&timeframe=${chartTimeframe}`);
      ws.onopen = () => { wsActive = true; };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.bars) {
            const candles = msg.bars.map(bar => ({
              time: Math.floor(new Date(bar.timestamp).getTime() / 1000),
              open: bar.open,
              high: bar.high,
              low: bar.low,
              close: bar.close,
              value: bar.close,
              volume: bar.volume
            }));
            setData({
              candles,
              line: candles.map(c => ({ time: c.time, value: c.close }))
            });
          }
          if (msg.indicators) {
            setIndicators(msg.indicators);
          }
          setLoading(false);
        } catch (e) {
          setError('Failed to parse real-time data');
        }
      };
      ws.onerror = (e) => {
        ws.close();
      };
      ws.onclose = () => {
        wsActive = false;
        if (!reconnectTimer) {
          reconnectTimer = setTimeout(connectWS, 2000); // Try reconnect in 2s
        }
      };
    }
    connectWS();
    // Fallback to REST fetch if WebSocket not available after 3s
    const fallbackTimeout = setTimeout(() => {
      if (!wsActive) {
        Promise.all([
          fetch(`${API_URL}/api/signal/markers?symbol=${symbol}&timeframe=${chartTimeframe}`).then(async res => { if (!res.ok) { let text = await res.text(); try { text = JSON.parse(text); } catch {} throw new Error(text.detail || text || res.status); } return res.json(); }).catch(e => { setError('Error loading markers: ' + e.message); return []; }),
          fetch(`${API_URL}/api/signal/indicators`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, timeframe: chartTimeframe })
          }).then(res => res.json()),
          fetch(`${API_URL}/api/ohlcv?symbol=${symbol}&timeframe=${chartTimeframe}`).then(async res => { if (!res.ok) { let text = await res.text(); try { text = JSON.parse(text); } catch {} throw new Error(text.detail || text || res.status); } return res.json(); }).catch(e => { setError('Error loading OHLCV: ' + e.message); return []; })
        ]).then(([markersData, indicatorsData, ohlcvRaw]) => {
          setTradeMarkers(markersData.markers || []);
          setIndicators(indicatorsData.indicators || {});
          if (!ohlcvRaw || !ohlcvRaw.data) throw new Error('No OHLCV data');
          const candles = ohlcvRaw.data.map(bar => ({
            time: Math.floor(new Date(bar.timestamp).getTime() / 1000),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            value: bar.close,
            volume: bar.volume
          }));
          setData({
            candles,
            line: candles.map(c => ({ time: c.time, value: c.close }))
          });
          setLoading(false);
        }).catch(e => {
          setError(e.message || 'Failed to load chart data');
          setLoading(false);
        });
      }
    }, 3000);
    return () => {
      if (ws) ws.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
      clearTimeout(fallbackTimeout);
    };
  }, [symbol, chartTimeframe]);

  // Robust chart initialization and update
  // --- Chart rendering logic (refactored for real-time support) ---
  const updateChart = () => {
    if (!data || !chartContainerRef.current) return;
    if (chartRef.current) chartRef.current.remove();
    chartRef.current = createChart(chartContainerRef.current, {
      height: 420,
      width: chartContainerRef.current.clientWidth,
      layout: { background: { type: 'Solid', color: '#181A20' }, textColor: '#DDD' },
      grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
      timeScale: { borderColor: '#333' },
      rightPriceScale: { borderColor: '#333' },
      crosshair: { mode: 1 }
    });
    if (chartType === 'candlestick') {
      candleSeriesRef.current = chartRef.current.addCandlestickSeries();
      candleSeriesRef.current.setData(data.candles);
      // Trade markers
      if (tradeMarkers && tradeMarkers.length > 0) {
        candleSeriesRef.current.setMarkers(tradeMarkers.map(marker => ({
          time: Math.floor(new Date(marker.timestamp).getTime() / 1000),
          position: marker.type === 'entry' ? 'belowBar' : 'aboveBar',
          color: marker.type === 'entry' ? 'green' : marker.type === 'take_profit' ? 'orange' : 'red',
          shape: marker.type === 'entry' ? 'arrowUp' : marker.type === 'take_profit' ? 'circle' : 'arrowDown',
          text: marker.type.replace('_', ' ').toUpperCase()
        })));
      }
      // Indicator overlays
      Object.entries(indicatorToggles).forEach(([key, enabled]) => {
        if (enabled && indicators[key]) {
          let color = '#FFD700';
          if (key === 'RSI') color = '#00BFFF';
          if (key === 'MACD_Strength') color = '#FF69B4';
          if (key === 'VWAP_Score') color = '#ADFF2F';
          if (key === 'Volume_Spike') color = '#FF8C00';
          const lineSeries = chartRef.current.addLineSeries({ color, lineWidth: 2 });
          let seriesData;
          if (Array.isArray(indicators[key])) {
            seriesData = data.candles.map((c, i) => ({ time: c.time, value: indicators[key][i] }));
          } else {
            seriesData = data.candles.map(c => ({ time: c.time, value: indicators[key] }));
          }
          lineSeries.setData(seriesData);
        }
      });
      // Trendlines
      trendlines.forEach(line => {
        const series = chartRef.current.addLineSeries({ color: '#90ee90', lineWidth: 2, priceLineVisible: false });
        series.setData([
          { time: line.start.time, value: line.start.price },
          { time: line.end.time, value: line.end.price }
        ]);
        series.applyOptions({ lastValueVisible: false, crosshairMarkerVisible: false });
      });
      // Annotations
      annotations.forEach(ann => {
        // Use price marker for annotation
        candleSeriesRef.current.createPriceLine({
          price: ann.price,
          color: '#FFA500',
          lineWidth: 1,
          lineStyle: 0,
          axisLabelVisible: true,
          title: ann.text
        });
      });
    } else {
      lineSeriesRef.current = chartRef.current.addLineSeries();
      lineSeriesRef.current.setData(data.line);
    }
  };

  useEffect(() => {
    updateChart();
    // eslint-disable-next-line
  }, [data, chartType, tradeMarkers, indicators, indicatorToggles, trendlines, annotations]);

  // --- Drawing tools event handlers ---
  // Trendline drawing
  useEffect(() => {
    if (!chartRef.current || drawingMode !== 'trendline') return;
    const chart = chartRef.current;
    const handleClick = (param) => {
      if (!param || !param.point) return;
      const price = chart.priceScale().coordinateToPrice(param.point.y);
      const time = chart.timeScale().coordinateToTime(param.point.x);
      if (!price || !time) return;
      if (!pendingLine) {
        setPendingLine({ start: { time, price } });
      } else {
        setTrendlines(lines => [...lines, { start: pendingLine.start, end: { time, price } }]);
        setPendingLine(null);
        setDrawingMode(null);
      }
    };
    chart.subscribeClick(handleClick);
    return () => chart.unsubscribeClick(handleClick);
  }, [drawingMode, pendingLine]);

  // Annotation drawing
  useEffect(() => {
    if (!chartRef.current || drawingMode !== 'annotation') return;
    const chart = chartRef.current;
    const handleClick = (param) => {
      if (!param || !param.point) return;
      const price = chart.priceScale().coordinateToPrice(param.point.y);
      const time = chart.timeScale().coordinateToTime(param.point.x);
      if (!price || !time) return;
      const text = window.prompt('Annotation label:');
      if (typeof text === 'string' && text.trim().length > 0) {
        setAnnotations(list => [...list, { time, price, text }]);
      }
      setDrawingMode(null);
    };
    chart.subscribeClick(handleClick);
    return () => chart.unsubscribeClick(handleClick);
  }, [drawingMode]);

  // Remove trendlines/annotations by click
  useEffect(() => {
    if (!chartRef.current || drawingMode) return;
    const chart = chartRef.current;
    const handleRemove = (param) => {
      if (!param || !param.point) return;
      const price = chart.priceScale().coordinateToPrice(param.point.y);
      const time = chart.timeScale().coordinateToTime(param.point.x);
      if (!price || !time) return;
      // Remove trendline if click is near
      setTrendlines(lines => lines.filter(line => {
        const dist = (p1, p2) => Math.sqrt(Math.pow(p1.time - p2.time, 2) + Math.pow(p1.price - p2.price, 2));
        const d1 = dist({ time, price }, line.start);
        const d2 = dist({ time, price }, line.end);
        return d1 > 0.01 && d2 > 0.01;
      }));
      // Remove annotation if click is near
      setAnnotations(list => list.filter(ann => {
        const dist = (p1, p2) => Math.sqrt(Math.pow(p1.time - p2.time, 2) + Math.pow(p1.price - p2.price, 2));
        return dist({ time, price }, ann) > 0.01;
      }));
    };
    chart.subscribeClick(handleRemove);
    return () => chart.unsubscribeClick(handleRemove);
  }, [drawingMode, trendlines, annotations]);

  /**
   * On mount, login to get token.
   * 
   * @description This effect is used to fetch the authentication token from the backend.
   */

  // Login handler fully removed for test mode


  /**
   * Fetch dashboard data when token is available and active tab is 'signals'.
   * 
   * @description This effect is used to fetch the dashboard data from the backend.
   */
  useEffect(() => {
    if (token && activeTab === 'signals') {
      // Add error handling for 401/403 (token expired)
      // Call a real protected endpoint to test token validity
      fetch(`${API_URL}/api/arbitrage/opportunities`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbol, min_spread: 0.01 })
      })
      .then(res => {
        if (res.status === 401 || res.status === 403) {
          
          window.location.href = '/login';
          return;
        }
        return res.json();
      })
      .then(data => {
        // ... handle arbitrage opportunities data
      })
      .catch(err => {
        if (err.message && err.message.includes('401')) {
          
          window.location.href = '/login';
        }
      });
      fetch(`${API_URL}/dashboard/${symbol}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => response.json())
        .then(data => setData(data))
        .catch(error => console.error('Error fetching dashboard data:', error));
    }
  }, [symbol, token, activeTab]);

  // If authenticated and admin, render admin panel
  // Show error message if any error is present (except for auth errors, which are bypassed)
  if (error && !error.toLowerCase().includes('auth')) {
    return (
      <ThemeProvider theme={createTheme()}>
        <CssBaseline />
        <Container maxWidth="sm" sx={{ mt: 8 }}>
          <Box className="card" sx={{ p: 4, textAlign: 'center', color: 'error.main' }}>
            <Typography variant="h5" color="error">Error</Typography>
            <Typography variant="body1">{error}</Typography>
            <Button variant="contained" color="primary" sx={{ mt: 2 }} onClick={() => window.location.reload()}>Reload</Button>
          </Box>
        </Container>
      </ThemeProvider>
    );
  }

  // Show loading indicator if loading
  if (loading) {
    return (
      <ThemeProvider theme={createTheme()}>
        <CssBaseline />
        <Container maxWidth="sm" sx={{ mt: 8 }}>
          <Box className="card" sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="h5">Loading...</Typography>
            <Typography variant="body2" color="textSecondary">Please wait while we load your dashboard.</Typography>
          </Box>
        </Container>
      </ThemeProvider>
    );
  }

  // Always render admin panel for testing if isAdmin is true
  if (isAdmin) {
    return (
      <ThemeProvider theme={createTheme()}>
        <CssBaseline />
        <ErrorBoundary>
          <AdminPanel user={user} />
        </ErrorBoundary>
      </ThemeProvider>
    );
  }

  // Always render dashboard for testing if not admin
  return (
    <ThemeProvider theme={createTheme()}>
      <CssBaseline />

      <AppBar position="static" color="primary" elevation={1}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            GoldenSignalsAI
          </Typography>
          <Button color="inherit" onClick={() => setActiveTab("signals")} startIcon={<DashboardIcon />} sx={{ mr: 1 }} variant={activeTab === "signals" ? "contained" : "text"}>
            Trading Signals
          </Button>
          <Button color="inherit" onClick={() => setActiveTab("arbitrage")} startIcon={<DashboardIcon />} sx={{ mr: 1 }} variant={activeTab === "arbitrage" ? "contained" : "text"}>
            Arbitrage
          </Button>
          <Button color="inherit" onClick={() => setActiveTab("admin")} startIcon={<AdminPanelSettingsIcon />} sx={{ mr: 1 }} variant={activeTab === "admin" ? "contained" : "text"}>
            Admin
          </Button>
        </Toolbar>
      </AppBar>
      <Container maxWidth="md" sx={{ mt: 4 }}>
        {activeTab === "signals" && (
          <Box className="card" sx={{ mb: 4 }}>
            <Typography variant="h5" gutterBottom>Trading Signals</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Typography variant="body1" sx={{ mr: 2 }}>Symbol:</Typography>
              <select onChange={(e) => setSymbol(e.target.value)} value={symbol} style={{ fontSize: 16, padding: 6, borderRadius: 6 }}>
                <option value="AAPL">AAPL</option>
                <option value="GOOGL">GOOGL</option>
                <option value="MSFT">MSFT</option>
              </select>
            </Box>
            {data ? (
              <Box>
                <Typography variant="h6">{data.symbol}</Typography>
                <Typography variant="body1">Price: ${data.price}</Typography>
                <Typography variant="body1">Trend: {data.trend}</Typography>
                <Box sx={{ mt: 4 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2 }}>Advanced Chart</Typography>
                  <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                    <Typography variant="body2">Timeframe:</Typography>
                    <select value={chartTimeframe} onChange={e => setChartTimeframe(e.target.value)} style={{ fontSize: 16, padding: 6, borderRadius: 6 }}>
                      <option value="1">1m</option>
                      <option value="5">5m</option>
                      <option value="15">15m</option>
                      <option value="30">30m</option>
                      <option value="60">1h</option>
                      <option value="D">1D</option>
                      <option value="W">1W</option>
                    </select>
                    <Typography variant="body2">Chart Type:</Typography>
                    <select value={chartType} onChange={e => setChartType(e.target.value)} style={{ fontSize: 16, padding: 6, borderRadius: 6 }}>
                      <option value="candlestick">Candlestick</option>
                      <option value="line">Line</option>
                    </select>
                  </Box>
                  {/* Indicator toggles */}
                  <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                    {Object.keys(indicatorToggles).map(ind => (
                      <label key={ind} style={{ marginRight: 10 }}>
                        <input type="checkbox" checked={indicatorToggles[ind]} onChange={() => setIndicatorToggles(toggles => ({ ...toggles, [ind]: !toggles[ind] }))} />
                        <span style={{ marginLeft: 4 }}>{ind.replace('_', ' ')}</span>
                      </label>
                    ))}
                  </Box>
                  <Box sx={{ width: '100%', height: 450 }}>
                    <div ref={chartContainerRef} style={{ height: 420, width: '100%' }} />
                  </Box>
                  {/* Placeholder for drawing tools */}
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant={drawingMode === 'trendline' ? "contained" : "outlined"}
                      size="small"
                      color={drawingMode === 'trendline' ? "primary" : "inherit"}
                      onClick={() => setDrawingMode(drawingMode === 'trendline' ? null : 'trendline')}
                    >
                      {drawingMode === 'trendline' ? 'Click Chart to Draw' : 'Draw Trendline'}
                    </Button>
                    <Button
                      variant={drawingMode === 'annotation' ? "contained" : "outlined"}
                      size="small"
                      color={drawingMode === 'annotation' ? "primary" : "inherit"}
                      sx={{ ml: 1 }}
                      onClick={() => setDrawingMode(drawingMode === 'annotation' ? null : 'annotation')}
                    >
                      {drawingMode === 'annotation' ? 'Click Chart to Add' : 'Add Annotation'}
                    </Button>
                  </Box>
                  {/* Trade markers from backend */}
                  <Box sx={{ mt: 1 }}>
                    {tradeMarkers.map((marker, idx) => (
                      <Typography key={idx} variant="caption" color={
                        marker.type === 'entry' ? 'success.main' :
                        marker.type === 'take_profit' ? 'warning.main' :
                        marker.type === 'exit' ? 'error.main' : 'text.secondary'
                      }>
                        {marker.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}: ${marker.price} ({new Date(marker.timestamp).toLocaleDateString()})
                      </Typography>
                    ))}
                  </Box>
                  {/* Technical indicators */}
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Technical Indicators:</Typography>
                    {Object.entries(indicators).length === 0 && (
                      <Typography variant="caption" color="text.secondary">Loading indicators...</Typography>
                    )}
                    {Object.entries(indicators).map(([key, value]) => (
                      <Typography key={key} variant="caption" sx={{ mr: 2 }}>
                        {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              </Box>
            ) : (
              <Typography variant="body2">Loading...</Typography>
            )}
          </Box>
        )}
        {activeTab === "arbitrage" && (
          <Box className="card">
            <Arbitrage />
          </Box>
        )}
        {activeTab === "admin" && (
          <Box className="card">
            <ErrorBoundary><AdminPanel /></ErrorBoundary>
          </Box>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;
