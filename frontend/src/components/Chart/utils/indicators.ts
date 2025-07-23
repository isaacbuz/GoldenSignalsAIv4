/**
 * Technical indicator calculations
 * Clean, efficient implementations
 */

import { CandlestickData, LineData, Time } from 'lightweight-charts';

export interface IndicatorResults {
  ma20?: LineData[];
  ma50?: LineData[];
  ma200?: LineData[];
  bollingerBands?: {
    upper: LineData[];
    middle: LineData[];
    lower: LineData[];
  };
  rsi?: LineData[];
  macd?: {
    macd: LineData[];
    signal: LineData[];
    histogram: LineData[];
  };
}

/**
 * Calculate all indicators for the given candle data
 */
export function calculateIndicators(candles: CandlestickData[]): IndicatorResults {
  return {
    ma20: calculateMA(candles, 20),
    ma50: calculateMA(candles, 50),
    ma200: calculateMA(candles, 200),
    bollingerBands: calculateBollingerBands(candles, 20, 2),
    rsi: calculateRSI(candles, 14),
    macd: calculateMACD(candles, 12, 26, 9),
  };
}

/**
 * Calculate Simple Moving Average
 */
export function calculateMA(candles: CandlestickData[], period: number): LineData[] {
  if (candles.length < period) return [];

  const ma: LineData[] = [];
  for (let i = period - 1; i < candles.length; i++) {
    const sum = candles.slice(i - period + 1, i + 1).reduce((acc, c) => acc + c.close, 0);
    ma.push({ time: candles[i].time, value: sum / period });
  }
  return ma;
}

/**
 * Calculate Exponential Moving Average
 */
export function calculateEMA(values: number[], period: number): number[] {
  if (values.length < period) return [];

  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += values[i];
  }
  ema[period - 1] = sum / period;

  // Calculate EMA
  for (let i = period; i < values.length; i++) {
    ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
  }

  return ema;
}

/**
 * Calculate Bollinger Bands
 */
export function calculateBollingerBands(
  candles: CandlestickData[],
  period: number,
  stdDev: number
) {
  const middle = calculateMA(candles, period);
  if (middle.length === 0) return undefined;

  const upper: LineData[] = [];
  const lower: LineData[] = [];

  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const avg = middle[i - period + 1].value;
    const squaredDiffs = slice.map(c => Math.pow(c.close - avg, 2));
    const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / period;
    const std = Math.sqrt(variance);

    upper.push({ time: candles[i].time, value: avg + std * stdDev });
    lower.push({ time: candles[i].time, value: avg - std * stdDev });
  }

  return { upper, middle, lower };
}

/**
 * Calculate RSI (Relative Strength Index)
 */
export function calculateRSI(candles: CandlestickData[], period: number): LineData[] {
  if (candles.length < period + 1) return [];

  const rsi: LineData[] = [];
  const gains: number[] = [];
  const losses: number[] = [];

  // Calculate initial gains/losses
  for (let i = 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? -change : 0);
  }

  // Calculate RSI
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b) / period;

  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;

    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsiValue = 100 - (100 / (1 + rs));
    rsi.push({ time: candles[i + 1].time, value: rsiValue });
  }

  return rsi;
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 */
export function calculateMACD(
  candles: CandlestickData[],
  fast: number,
  slow: number,
  signal: number
) {
  if (candles.length < slow) return undefined;

  const closes = candles.map(c => c.close);
  const ema12 = calculateEMA(closes, fast);
  const ema26 = calculateEMA(closes, slow);

  const macdLine: LineData[] = [];
  const macdValues: number[] = [];

  for (let i = slow - 1; i < candles.length; i++) {
    const macdValue = ema12[i] - ema26[i];
    macdValues.push(macdValue);
    macdLine.push({ time: candles[i].time, value: macdValue });
  }

  const signalEMA = calculateEMA(macdValues, signal);
  const signalLine: LineData[] = [];
  const histogram: LineData[] = [];

  for (let i = signal - 1; i < signalEMA.length; i++) {
    const time = candles[i + slow - 1].time;
    signalLine.push({ time, value: signalEMA[i] });
    histogram.push({ time, value: macdValues[i] - signalEMA[i] });
  }

  return { macd: macdLine, signal: signalLine, histogram };
}

/**
 * Calculate Volume Weighted Average Price (VWAP)
 */
export function calculateVWAP(candles: CandlestickData[], volumes: number[]): LineData[] {
  const vwap: LineData[] = [];
  let cumulativeTPV = 0;
  let cumulativeVolume = 0;

  for (let i = 0; i < candles.length; i++) {
    const typicalPrice = (candles[i].high + candles[i].low + candles[i].close) / 3;
    cumulativeTPV += typicalPrice * volumes[i];
    cumulativeVolume += volumes[i];

    vwap.push({
      time: candles[i].time,
      value: cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice
    });
  }

  return vwap;
}

/**
 * Calculate ATR (Average True Range)
 */
export function calculateATR(candles: CandlestickData[], period: number): LineData[] {
  if (candles.length < period + 1) return [];

  const tr: number[] = [];

  // Calculate True Range
  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;

    tr.push(Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    ));
  }

  // Calculate ATR
  const atr: LineData[] = [];
  let atrValue = tr.slice(0, period).reduce((a, b) => a + b) / period;

  for (let i = period; i < tr.length; i++) {
    atrValue = (atrValue * (period - 1) + tr[i]) / period;
    atr.push({ time: candles[i + 1].time, value: atrValue });
  }

  return atr;
}
