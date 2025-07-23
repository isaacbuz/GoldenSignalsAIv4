/**
 * Simple candle normalizer to fix unrealistic candlestick sizes
 * This addresses the root cause without over-engineering
 */

import { ChartDataPoint } from '../types/chart';
import logger from '../services/logger';


interface NormalizationConfig {
  // Maximum allowed range as percentage of price
  maxRangePercent: Record<string, number>;
  // Volume outlier detection
  volumeOutlierMultiplier: number;
}

const DEFAULT_CONFIG: NormalizationConfig = {
  maxRangePercent: {
    '1m': 2.0,    // 2% for 1 minute (real market can move this much)
    '5m': 3.0,    // 3% for 5 minutes
    '15m': 4.0,   // 4% for 15 minutes
    '30m': 5.0,   // 5% for 30 minutes
    '1h': 7.0,    // 7% for 1 hour
    '4h': 10.0,   // 10% for 4 hours
    '1d': 15.0,   // 15% for 1 day (stocks can easily move 10%+ in a day)
    '1w': 25.0,   // 25% for 1 week
    '1M': 50.0,   // 50% for 1 month
  },
  volumeOutlierMultiplier: 20, // Increased to handle volume spikes
};

export class CandleNormalizer {
  private config: NormalizationConfig;
  private warnings: string[] = [];

  constructor(config: Partial<NormalizationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Normalize a single candle
   */
  normalizeCandle(candle: ChartDataPoint, timeframe: string): ChartDataPoint {
    const maxRangePercent = this.config.maxRangePercent[timeframe] || 1.0;
    const avgPrice = (candle.high + candle.low) / 2;
    const currentRange = candle.high - candle.low;
    const maxAllowedRange = avgPrice * (maxRangePercent / 100);

    // Check if candle exceeds maximum allowed range
    if (currentRange > maxAllowedRange) {
      const scaleFactor = maxAllowedRange / currentRange;

      // Log warning for debugging
      this.warnings.push(
        `Large candle detected at ${new Date(candle.time * 1000).toISOString()}: ` +
        `${((currentRange / avgPrice) * 100).toFixed(2)}% range (max: ${maxRangePercent}%)`
      );

      // Scale down the candle proportionally
      const midPrice = (candle.high + candle.low) / 2;

      return {
        ...candle,
        high: midPrice + (candle.high - midPrice) * scaleFactor,
        low: midPrice - (midPrice - candle.low) * scaleFactor,
        open: midPrice + (candle.open - midPrice) * scaleFactor,
        close: midPrice + (candle.close - midPrice) * scaleFactor,
      };
    }

    return candle;
  }

  /**
   * Normalize an array of candles
   */
  normalizeData(data: ChartDataPoint[], timeframe: string): ChartDataPoint[] {
    this.warnings = [];

    // Calculate average volume for outlier detection
    const avgVolume = data.reduce((sum, d) => sum + d.volume, 0) / data.length;
    const maxVolume = avgVolume * this.config.volumeOutlierMultiplier;

    const normalized = data.map(candle => {
      let normalizedCandle = this.normalizeCandle(candle, timeframe);

      // Also normalize volume outliers
      if (normalizedCandle.volume > maxVolume) {
        this.warnings.push(
          `Volume outlier at ${new Date(candle.time * 1000).toISOString()}: ` +
          `${candle.volume.toLocaleString()} (avg: ${avgVolume.toFixed(0)})`
        );
        normalizedCandle = {
          ...normalizedCandle,
          volume: maxVolume,
        };
      }

      return normalizedCandle;
    });

    // Log warnings if any
    if (this.warnings.length > 0) {
      logger.warn('🚨 Data normalization warnings:', this.warnings);
    }

    return normalized;
  }

  /**
   * Check if data needs normalization
   */
  needsNormalization(data: ChartDataPoint[], timeframe: string): boolean {
    const maxRangePercent = this.config.maxRangePercent[timeframe] || 1.0;

    return data.some(candle => {
      const avgPrice = (candle.high + candle.low) / 2;
      const range = candle.high - candle.low;
      const rangePercent = (range / avgPrice) * 100;
      return rangePercent > maxRangePercent;
    });
  }

  /**
   * Get normalization statistics
   */
  getStats(data: ChartDataPoint[]): {
    avgRange: number;
    maxRange: number;
    outliers: number;
    avgVolume: number;
  } {
    const ranges = data.map(d => {
      const avg = (d.high + d.low) / 2;
      return ((d.high - d.low) / avg) * 100;
    });

    const volumes = data.map(d => d.volume);
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

    return {
      avgRange: ranges.reduce((a, b) => a + b, 0) / ranges.length,
      maxRange: Math.max(...ranges),
      outliers: ranges.filter(r => r > 2).length,
      avgVolume,
    };
  }
}

// Singleton instance
export const candleNormalizer = new CandleNormalizer();

// Helper function for quick normalization
export const normalizeChartData = (
  data: ChartDataPoint[],
  timeframe: string
): ChartDataPoint[] => {
  return candleNormalizer.normalizeData(data, timeframe);
};
