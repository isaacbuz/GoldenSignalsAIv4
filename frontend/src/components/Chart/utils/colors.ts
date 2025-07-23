/**
 * Chart color constants
 * Professional color scheme for trading
 */

export const CHART_COLORS = {
  // Background colors
  background: '#131722',
  card: '#1e222d',
  border: '#2a2e39',

  // Text colors
  text: {
    primary: '#d1d4dc',
    secondary: '#787b86',
  },

  // Candlestick colors
  candle: {
    up: '#26a69a',
    down: '#ef5350',
  },

  // Moving average colors
  ma: {
    20: '#2962ff',
    50: '#ff6d00',
    200: '#d500f9',
  },

  // Volume colors
  volume: {
    up: 'rgba(38, 166, 154, 0.5)',
    down: 'rgba(239, 83, 80, 0.5)',
  },

  // Grid color
  grid: 'rgba(42, 46, 57, 0.5)',

  // Bollinger Bands
  bollinger: '#607d8b',

  // RSI colors
  rsi: {
    line: '#ab47bc',
    overbought: '#ef5350',
    oversold: '#26a69a',
  },

  // MACD colors
  macd: {
    line: '#2196f3',
    signal: '#ff6d00',
    histogram: {
      positive: '#26a69a',
      negative: '#ef5350',
    },
  },

  // AI/Agent colors
  ai: {
    buy: '#26a69a',
    sell: '#ef5350',
    hold: '#ffc107',
    analyzing: '#2196f3',
  },

  // Drawing tools
  drawing: {
    line: '#2196f3',
    fibonacci: '#ff9800',
    rectangle: '#4caf50',
    text: '#ffffff',
  },
};
