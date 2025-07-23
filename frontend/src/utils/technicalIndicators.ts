// Technical Indicators Calculation Library
import { Time } from 'lightweight-charts';

export interface OHLCData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface IndicatorValue {
  time: Time;
  value: number;
}

export interface BollingerBandsValue {
  time: Time;
  upper: number;
  middle: number;
  lower: number;
}

export interface MACDValue {
  time: Time;
  macd: number;
  signal: number;
  histogram: number;
}

// Simple Moving Average (SMA)
export function calculateSMA(data: OHLCData[], period: number): IndicatorValue[] {
  const result: IndicatorValue[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push({ time: data[i].time, value: NaN });
    } else {
      const sum = data.slice(i - period + 1, i + 1)
        .reduce((acc, candle) => acc + candle.close, 0);
      result.push({ time: data[i].time, value: sum / period });
    }
  }

  return result;
}

// Exponential Moving Average (EMA)
export function calculateEMA(data: OHLCData[], period: number): IndicatorValue[] {
  const result: IndicatorValue[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA for the first value
  const sma = data.slice(0, period).reduce((acc, candle) => acc + candle.close, 0) / period;
  result.push({ time: data[period - 1].time, value: sma });

  // Calculate EMA for remaining values
  for (let i = period; i < data.length; i++) {
    const ema = (data[i].close - result[result.length - 1].value) * multiplier + result[result.length - 1].value;
    result.push({ time: data[i].time, value: ema });
  }

  // Fill initial values with NaN
  for (let i = 0; i < period - 1; i++) {
    result.unshift({ time: data[i].time, value: NaN });
  }

  return result;
}

// Relative Strength Index (RSI)
export function calculateRSI(data: OHLCData[], period: number = 14): IndicatorValue[] {
  const result: IndicatorValue[] = [];
  const gains: number[] = [];
  const losses: number[] = [];

  // Calculate price changes
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close;
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }

  // Calculate initial average gain/loss
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

  // First RSI value
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  result.push({ time: data[period].time, value: rsi });

  // Calculate subsequent RSI values
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;

    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    result.push({ time: data[i + 1].time, value: rsi });
  }

  // Fill initial values with NaN
  for (let i = 0; i <= period; i++) {
    result.unshift({ time: data[i].time, value: NaN });
  }

  return result;
}

// MACD (Moving Average Convergence Divergence)
export function calculateMACD(
  data: OHLCData[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACDValue[] {
  const result: MACDValue[] = [];

  const ema12 = calculateEMA(data, fastPeriod);
  const ema26 = calculateEMA(data, slowPeriod);

  // Calculate MACD line
  const macdLine: IndicatorValue[] = [];
  for (let i = 0; i < data.length; i++) {
    if (isNaN(ema12[i].value) || isNaN(ema26[i].value)) {
      macdLine.push({ time: data[i].time, value: NaN });
    } else {
      macdLine.push({ time: data[i].time, value: ema12[i].value - ema26[i].value });
    }
  }

  // Calculate signal line (EMA of MACD)
  const signalLine = calculateEMAFromValues(macdLine, signalPeriod);

  // Calculate histogram
  for (let i = 0; i < data.length; i++) {
    if (isNaN(macdLine[i].value) || isNaN(signalLine[i].value)) {
      result.push({
        time: data[i].time,
        macd: NaN,
        signal: NaN,
        histogram: NaN,
      });
    } else {
      result.push({
        time: data[i].time,
        macd: macdLine[i].value,
        signal: signalLine[i].value,
        histogram: macdLine[i].value - signalLine[i].value,
      });
    }
  }

  return result;
}

// Bollinger Bands
export function calculateBollingerBands(
  data: OHLCData[],
  period: number = 20,
  stdDev: number = 2
): BollingerBandsValue[] {
  const result: BollingerBandsValue[] = [];
  const sma = calculateSMA(data, period);

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push({
        time: data[i].time,
        upper: NaN,
        middle: NaN,
        lower: NaN,
      });
    } else {
      const slice = data.slice(i - period + 1, i + 1);
      const mean = sma[i].value;

      // Calculate standard deviation
      const squaredDiffs = slice.map(candle => Math.pow(candle.close - mean, 2));
      const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
      const standardDeviation = Math.sqrt(variance);

      result.push({
        time: data[i].time,
        upper: mean + (standardDeviation * stdDev),
        middle: mean,
        lower: mean - (standardDeviation * stdDev),
      });
    }
  }

  return result;
}

// Stochastic Oscillator
export function calculateStochastic(
  data: OHLCData[],
  period: number = 14,
  smoothK: number = 3,
  smoothD: number = 3
): { k: IndicatorValue[], d: IndicatorValue[] } {
  const kValues: IndicatorValue[] = [];

  // Calculate %K
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      kValues.push({ time: data[i].time, value: NaN });
    } else {
      const slice = data.slice(i - period + 1, i + 1);
      const highs = slice.map(candle => candle.high);
      const lows = slice.map(candle => candle.low);

      const highest = Math.max(...highs);
      const lowest = Math.min(...lows);
      const current = data[i].close;

      const k = ((current - lowest) / (highest - lowest)) * 100;
      kValues.push({ time: data[i].time, value: k });
    }
  }

  // Smooth %K
  const smoothedK = calculateSMAFromValues(kValues, smoothK);

  // Calculate %D (SMA of smoothed %K)
  const dValues = calculateSMAFromValues(smoothedK, smoothD);

  return { k: smoothedK, d: dValues };
}

// ATR (Average True Range)
export function calculateATR(data: OHLCData[], period: number = 14): IndicatorValue[] {
  const result: IndicatorValue[] = [];
  const trueRanges: number[] = [];

  // Calculate True Range
  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      trueRanges.push(data[i].high - data[i].low);
    } else {
      const highLow = data[i].high - data[i].low;
      const highClose = Math.abs(data[i].high - data[i - 1].close);
      const lowClose = Math.abs(data[i].low - data[i - 1].close);

      trueRanges.push(Math.max(highLow, highClose, lowClose));
    }
  }

  // Calculate ATR
  let atr = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push({ time: data[period - 1].time, value: atr });

  for (let i = period; i < data.length; i++) {
    atr = ((atr * (period - 1)) + trueRanges[i]) / period;
    result.push({ time: data[i].time, value: atr });
  }

  // Fill initial values with NaN
  for (let i = 0; i < period - 1; i++) {
    result.unshift({ time: data[i].time, value: NaN });
  }

  return result;
}

// ADX (Average Directional Index)
export interface ADXValue {
  time: Time;
  adx: number;
  plusDI: number;
  minusDI: number;
}

export function calculateADX(data: OHLCData[], period: number = 14): ADXValue[] {
  const result: ADXValue[] = [];

  if (data.length < period + 1) {
    return result;
  }

  // Calculate True Range, +DM and -DM
  const trueRanges: number[] = [];
  const plusDMs: number[] = [];
  const minusDMs: number[] = [];

  for (let i = 1; i < data.length; i++) {
    // True Range
    const highLow = data[i].high - data[i].low;
    const highClose = Math.abs(data[i].high - data[i - 1].close);
    const lowClose = Math.abs(data[i].low - data[i - 1].close);
    const tr = Math.max(highLow, highClose, lowClose);
    trueRanges.push(tr);

    // Directional Movement
    const upMove = data[i].high - data[i - 1].high;
    const downMove = data[i - 1].low - data[i].low;

    const plusDM = upMove > downMove && upMove > 0 ? upMove : 0;
    const minusDM = downMove > upMove && downMove > 0 ? downMove : 0;

    plusDMs.push(plusDM);
    minusDMs.push(minusDM);
  }

  // Calculate smoothed values
  let smoothedTR = trueRanges.slice(0, period).reduce((a, b) => a + b, 0);
  let smoothedPlusDM = plusDMs.slice(0, period).reduce((a, b) => a + b, 0);
  let smoothedMinusDM = minusDMs.slice(0, period).reduce((a, b) => a + b, 0);

  const dx: number[] = [];

  for (let i = period; i < trueRanges.length; i++) {
    // Smooth the values (Wilder's smoothing)
    smoothedTR = smoothedTR - (smoothedTR / period) + trueRanges[i];
    smoothedPlusDM = smoothedPlusDM - (smoothedPlusDM / period) + plusDMs[i];
    smoothedMinusDM = smoothedMinusDM - (smoothedMinusDM / period) + minusDMs[i];

    // Calculate +DI and -DI
    const plusDI = (smoothedPlusDM / smoothedTR) * 100;
    const minusDI = (smoothedMinusDM / smoothedTR) * 100;

    // Calculate DX
    const diSum = plusDI + minusDI;
    const diDiff = Math.abs(plusDI - minusDI);
    const dxValue = diSum === 0 ? 0 : (diDiff / diSum) * 100;
    dx.push(dxValue);

    // Store +DI and -DI for the first ADX calculation
    if (dx.length === 1) {
      result.push({
        time: data[i + 1].time,
        adx: NaN,
        plusDI,
        minusDI,
      });
    }
  }

  // Calculate ADX (average of DX)
  if (dx.length >= period) {
    let adx = dx.slice(0, period).reduce((a, b) => a + b, 0) / period;

    // Update the first ADX value
    const firstIndex = period * 2;
    if (firstIndex < data.length) {
      result[0] = {
        ...result[0],
        adx,
      };
    }

    // Calculate subsequent ADX values
    for (let i = period; i < dx.length; i++) {
      adx = ((adx * (period - 1)) + dx[i]) / period;

      const dataIndex = i + period + 1;
      if (dataIndex < data.length) {
        // Recalculate current +DI and -DI
        const currentTR = trueRanges[i];
        const currentPlusDM = plusDMs[i];
        const currentMinusDM = minusDMs[i];

        const smoothedTRCurrent = smoothedTR - (smoothedTR / period) + currentTR;
        const smoothedPlusDMCurrent = smoothedPlusDM - (smoothedPlusDM / period) + currentPlusDM;
        const smoothedMinusDMCurrent = smoothedMinusDM - (smoothedMinusDM / period) + currentMinusDM;

        const plusDI = (smoothedPlusDMCurrent / smoothedTRCurrent) * 100;
        const minusDI = (smoothedMinusDMCurrent / smoothedTRCurrent) * 100;

        result.push({
          time: data[dataIndex].time,
          adx,
          plusDI,
          minusDI,
        });
      }
    }
  }

  // Fill initial values with NaN
  const nanValues: ADXValue[] = [];
  for (let i = 0; i < Math.min(period * 2, data.length); i++) {
    if (i < data.length) {
      nanValues.push({
        time: data[i].time,
        adx: NaN,
        plusDI: NaN,
        minusDI: NaN,
      });
    }
  }

  return [...nanValues, ...result];
}

// VWAP (Volume Weighted Average Price)
export function calculateVWAP(data: OHLCData[]): IndicatorValue[] {
  const result: IndicatorValue[] = [];
  let cumulativeVolume = 0;
  let cumulativeVolumePrice = 0;

  // Reset at the start of each trading day
  let lastDate = new Date((data[0].time as number) * 1000).getDate();

  for (let i = 0; i < data.length; i++) {
    const currentDate = new Date((data[i].time as number) * 1000).getDate();

    // Reset if new day
    if (currentDate !== lastDate) {
      cumulativeVolume = 0;
      cumulativeVolumePrice = 0;
      lastDate = currentDate;
    }

    const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
    const volume = data[i].volume || 0;

    cumulativeVolumePrice += typicalPrice * volume;
    cumulativeVolume += volume;

    const vwap = cumulativeVolume > 0 ? cumulativeVolumePrice / cumulativeVolume : typicalPrice;

    result.push({ time: data[i].time, value: vwap });
  }

  return result;
}

// Helper function to calculate EMA from values
function calculateEMAFromValues(values: IndicatorValue[], period: number): IndicatorValue[] {
  const result: IndicatorValue[] = [];
  const multiplier = 2 / (period + 1);

  // Find first non-NaN value
  let firstValidIndex = values.findIndex(v => !isNaN(v.value));
  if (firstValidIndex === -1) return values.map(v => ({ time: v.time, value: NaN }));

  // Calculate SMA for first valid values
  let validCount = 0;
  let sum = 0;
  let smaIndex = -1;

  for (let i = firstValidIndex; i < values.length && validCount < period; i++) {
    if (!isNaN(values[i].value)) {
      sum += values[i].value;
      validCount++;
      if (validCount === period) {
        smaIndex = i;
      }
    }
  }

  if (smaIndex === -1) return values.map(v => ({ time: v.time, value: NaN }));

  // Fill with NaN up to SMA index
  for (let i = 0; i <= smaIndex; i++) {
    if (i < smaIndex) {
      result.push({ time: values[i].time, value: NaN });
    } else {
      result.push({ time: values[i].time, value: sum / period });
    }
  }

  // Calculate EMA for remaining values
  for (let i = smaIndex + 1; i < values.length; i++) {
    if (isNaN(values[i].value)) {
      result.push({ time: values[i].time, value: NaN });
    } else {
      const ema = (values[i].value - result[result.length - 1].value) * multiplier + result[result.length - 1].value;
      result.push({ time: values[i].time, value: ema });
    }
  }

  return result;
}

// Helper function to calculate SMA from values
function calculateSMAFromValues(values: IndicatorValue[], period: number): IndicatorValue[] {
  const result: IndicatorValue[] = [];

  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      result.push({ time: values[i].time, value: NaN });
    } else {
      const slice = values.slice(i - period + 1, i + 1);
      const validValues = slice.filter(v => !isNaN(v.value));

      if (validValues.length === period) {
        const sum = validValues.reduce((acc, v) => acc + v.value, 0);
        result.push({ time: values[i].time, value: sum / period });
      } else {
        result.push({ time: values[i].time, value: NaN });
      }
    }
  }

  return result;
}

// Pattern Detection
export interface Pattern {
  type: 'triangle' | 'flag' | 'pennant' | 'head_shoulders' | 'double_top' | 'double_bottom';
  startIndex: number;
  endIndex: number;
  confidence: number;
  trendLines?: { start: { time: Time; price: number }, end: { time: Time; price: number } }[];
  target?: number;
}

// Simple pattern detection - this would be much more complex in production
export function detectPatterns(data: OHLCData[]): Pattern[] {
  const patterns: Pattern[] = [];

  // Detect potential triangles
  const triangles = detectTriangles(data);
  patterns.push(...triangles);

  // Add more pattern detection logic here

  return patterns;
}

function detectTriangles(data: OHLCData[]): Pattern[] {
  const patterns: Pattern[] = [];
  const minPatternLength = 20;

  // Simple triangle detection logic
  for (let i = minPatternLength; i < data.length - 5; i++) {
    const slice = data.slice(i - minPatternLength, i);

    // Calculate highs and lows
    const highs = slice.map((candle, index) => ({ index: i - minPatternLength + index, value: candle.high }));
    const lows = slice.map((candle, index) => ({ index: i - minPatternLength + index, value: candle.low }));

    // Check for converging trendlines
    const highTrend = calculateTrendline(highs);
    const lowTrend = calculateTrendline(lows);

    if (highTrend && lowTrend) {
      const convergence = Math.abs(highTrend.slope) + Math.abs(lowTrend.slope);

      // If trends are converging
      if (convergence < 0.1 && highTrend.slope < 0 && lowTrend.slope > 0) {
        patterns.push({
          type: 'triangle',
          startIndex: i - minPatternLength,
          endIndex: i,
          confidence: 0.7,
          trendLines: [
            {
              start: { time: data[i - minPatternLength].time, price: highTrend.start },
              end: { time: data[i].time, price: highTrend.end }
            },
            {
              start: { time: data[i - minPatternLength].time, price: lowTrend.start },
              end: { time: data[i].time, price: lowTrend.end }
            }
          ],
          target: data[i].close * 1.05 // Simple 5% target
        });
      }
    }
  }

  return patterns;
}

function calculateTrendline(points: { index: number; value: number }[]): { slope: number; start: number; end: number } | null {
  if (points.length < 2) return null;

  // Simple linear regression
  const n = points.length;
  const sumX = points.reduce((sum, p) => sum + p.index, 0);
  const sumY = points.reduce((sum, p) => sum + p.value, 0);
  const sumXY = points.reduce((sum, p) => sum + p.index * p.value, 0);
  const sumX2 = points.reduce((sum, p) => sum + p.index * p.index, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  return {
    slope,
    start: slope * points[0].index + intercept,
    end: slope * points[points.length - 1].index + intercept
  };
}
