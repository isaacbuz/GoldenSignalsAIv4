/**
 * Market Data Normalizer
 *
 * Ensures chart data looks realistic by:
 * - Filling gaps in data (weekends, holidays, missing candles)
 * - Adding market-appropriate volatility
 * - Handling market hours correctly
 * - Timeframe-specific normalization
 *
 * Based on Grok's recommendations for professional charting
 */

import { MarketDataPoint } from './backendMarketDataService';
import logger from './logger';

interface NormalizedData extends MarketDataPoint {
  isGap?: boolean;
  isAfterHours?: boolean;
  isPreMarket?: boolean;
}

export class MarketDataNormalizer {
  // Market hours in EST/EDT
  private readonly MARKET_OPEN = 9.5; // 9:30 AM
  private readonly MARKET_CLOSE = 16; // 4:00 PM
  private readonly PRE_MARKET_START = 4; // 4:00 AM
  private readonly AFTER_HOURS_END = 20; // 8:00 PM

  /**
   * Normalize and fill gaps in market data based on timeframe
   */
  normalizeData(
    data: MarketDataPoint[],
    timeframe: string,
    options: {
      fillGaps?: boolean;
      addVolatility?: boolean;
      markMarketHours?: boolean;
    } = {}
  ): NormalizedData[] {
    const { fillGaps = true, addVolatility = false, markMarketHours = true } = options;

    if (!data || data.length === 0) return [];

    // Sort data by time
    const sortedData = [...data].sort((a, b) =>
      (typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime() / 1000) -
      (typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime() / 1000)
    );

    let normalized: NormalizedData[] = [];

    // Process based on timeframe
    if (this.isIntradayTimeframe(timeframe)) {
      normalized = this.normalizeIntradayData(sortedData, timeframe, fillGaps);
    } else {
      normalized = this.normalizeDailyData(sortedData, timeframe, fillGaps);
    }

    // Add realistic volatility if requested (for testing/demo)
    if (addVolatility) {
      normalized = this.addRealisticVolatility(normalized, timeframe);
    }

    // Mark market hours
    if (markMarketHours && this.isIntradayTimeframe(timeframe)) {
      normalized = this.markMarketHours(normalized);
    }

    return normalized;
  }

  /**
   * Check if timeframe is intraday
   */
  private isIntradayTimeframe(timeframe: string): boolean {
    return ['1m', '5m', '15m', '30m', '1h', '4h'].includes(timeframe);
  }

  /**
   * Normalize intraday data with gap filling
   */
  private normalizeIntradayData(
    data: MarketDataPoint[],
    timeframe: string,
    fillGaps: boolean
  ): NormalizedData[] {
    if (!fillGaps) return data as NormalizedData[];

    const intervalMinutes = this.getIntervalMinutes(timeframe);
    const normalized: NormalizedData[] = [];

    for (let i = 0; i < data.length; i++) {
      const current = data[i];
      const next = data[i + 1];

      // Add current candle
      normalized.push(current as NormalizedData);

      // Check for gap
      if (next) {
        const currentTime = this.getTimestamp(current.time);
        const nextTime = this.getTimestamp(next.time);
        const expectedNextTime = currentTime + (intervalMinutes * 60);

        // If gap is larger than expected interval
        if (nextTime - currentTime > intervalMinutes * 60 * 1.5) {
          // Fill gap with flat candles
          const gapCandles = this.fillIntradayGap(
            current,
            next,
            intervalMinutes,
            currentTime,
            nextTime
          );
          normalized.push(...gapCandles);
        }
      }
    }

    return normalized;
  }

  /**
   * Fill intraday gaps with appropriate candles
   */
  private fillIntradayGap(
    lastCandle: MarketDataPoint,
    nextCandle: MarketDataPoint,
    intervalMinutes: number,
    startTime: number,
    endTime: number
  ): NormalizedData[] {
    const gaps: NormalizedData[] = [];
    let currentTime = startTime + (intervalMinutes * 60);

    while (currentTime < endTime) {
      const date = new Date(currentTime * 1000);
      const hour = date.getHours();
      const isWeekend = date.getDay() === 0 || date.getDay() === 6;

      // Skip weekends for intraday
      if (!isWeekend) {
        // Check if within market hours
        const isMarketHours = hour >= 9.5 && hour < 16;
        const isExtendedHours = (hour >= 4 && hour < 9.5) || (hour >= 16 && hour < 20);

        if (isMarketHours || isExtendedHours) {
          gaps.push({
            time: currentTime,
            open: lastCandle.close,
            high: lastCandle.close,
            low: lastCandle.close,
            close: lastCandle.close,
            volume: 0,
            isGap: true,
            isPreMarket: hour >= 4 && hour < 9.5,
            isAfterHours: hour >= 16 && hour < 20,
          });
        }
      }

      currentTime += intervalMinutes * 60;
    }

    return gaps;
  }

  /**
   * Normalize daily/weekly/monthly data
   */
  private normalizeDailyData(
    data: MarketDataPoint[],
    timeframe: string,
    fillGaps: boolean
  ): NormalizedData[] {
    if (!fillGaps) return data as NormalizedData[];

    const normalized: NormalizedData[] = [];
    const intervalDays = this.getIntervalDays(timeframe);

    for (let i = 0; i < data.length; i++) {
      const current = data[i];
      const next = data[i + 1];

      normalized.push(current as NormalizedData);

      if (next) {
        const currentTime = this.getTimestamp(current.time);
        const nextTime = this.getTimestamp(next.time);
        const daysDiff = (nextTime - currentTime) / (24 * 60 * 60);

        // Fill weekends/holidays for daily data
        if (timeframe === '1d' && daysDiff > 1.5) {
          const gapDays = Math.floor(daysDiff);
          for (let d = 1; d < gapDays; d++) {
            const gapTime = currentTime + (d * 24 * 60 * 60);
            const gapDate = new Date(gapTime * 1000);

            // Skip weekends
            if (gapDate.getDay() !== 0 && gapDate.getDay() !== 6) {
              normalized.push({
                time: gapTime,
                open: current.close,
                high: current.close,
                low: current.close,
                close: current.close,
                volume: 0,
                isGap: true,
              });
            }
          }
        }
      }
    }

    return normalized;
  }

  /**
   * Add realistic volatility to flat/gap candles
   */
  private addRealisticVolatility(
    data: NormalizedData[],
    timeframe: string
  ): NormalizedData[] {
    const volatility = this.getTimeframeVolatility(timeframe);

    return data.map((candle, index) => {
      // Only add volatility to gap candles or flat candles
      if (candle.isGap || (candle.high === candle.low)) {
        const basePrice = candle.close;
        const randomFactor = (Math.random() - 0.5) * 2;
        const priceMove = basePrice * (volatility / 100) * randomFactor;

        const open = basePrice + priceMove * 0.3;
        const close = basePrice + priceMove;
        const high = Math.max(open, close) + Math.abs(priceMove * 0.2);
        const low = Math.min(open, close) - Math.abs(priceMove * 0.2);

        // Add some volume based on surrounding candles
        const avgVolume = this.getAverageVolume(data, index, 5);
        const volume = candle.isGap
          ? Math.floor(avgVolume * 0.3 * (0.5 + Math.random()))
          : candle.volume || avgVolume;

        return {
          ...candle,
          open: Number(open.toFixed(2)),
          high: Number(high.toFixed(2)),
          low: Number(low.toFixed(2)),
          close: Number(close.toFixed(2)),
          volume: Math.floor(volume),
        };
      }

      return candle;
    });
  }

  /**
   * Mark market hours for intraday data
   */
  private markMarketHours(data: NormalizedData[]): NormalizedData[] {
    return data.map(candle => {
      const date = new Date(this.getTimestamp(candle.time) * 1000);
      const hour = date.getHours() + date.getMinutes() / 60;

      return {
        ...candle,
        isPreMarket: hour >= this.PRE_MARKET_START && hour < this.MARKET_OPEN,
        isAfterHours: hour >= this.MARKET_CLOSE && hour < this.AFTER_HOURS_END,
      };
    });
  }

  /**
   * Get interval in minutes for timeframe
   */
  private getIntervalMinutes(timeframe: string): number {
    const map: Record<string, number> = {
      '1m': 1,
      '5m': 5,
      '15m': 15,
      '30m': 30,
      '1h': 60,
      '4h': 240,
    };
    return map[timeframe] || 60;
  }

  /**
   * Get interval in days for timeframe
   */
  private getIntervalDays(timeframe: string): number {
    const map: Record<string, number> = {
      '1d': 1,
      '1w': 7,
      '1M': 30,
    };
    return map[timeframe] || 1;
  }

  /**
   * Get typical volatility for timeframe
   */
  private getTimeframeVolatility(timeframe: string): number {
    const map: Record<string, number> = {
      '1m': 0.05,   // 0.05% for 1 minute
      '5m': 0.1,    // 0.1% for 5 minutes
      '15m': 0.2,   // 0.2% for 15 minutes
      '30m': 0.3,   // 0.3% for 30 minutes
      '1h': 0.5,    // 0.5% for 1 hour
      '4h': 1.0,    // 1% for 4 hours
      '1d': 2.0,    // 2% for 1 day
      '1w': 5.0,    // 5% for 1 week
      '1M': 10.0,   // 10% for 1 month
    };
    return map[timeframe] || 0.5;
  }

  /**
   * Convert time to timestamp
   */
  private getTimestamp(time: string | number): number {
    return typeof time === 'number' ? time : new Date(time).getTime() / 1000;
  }

  /**
   * Get average volume around index
   */
  private getAverageVolume(data: NormalizedData[], index: number, window: number): number {
    let sum = 0;
    let count = 0;

    for (let i = Math.max(0, index - window); i < Math.min(data.length, index + window); i++) {
      if (data[i].volume && !data[i].isGap) {
        sum += data[i].volume;
        count++;
      }
    }

    return count > 0 ? sum / count : 100000;
  }

  /**
   * Validate data quality
   */
  validateData(data: MarketDataPoint[]): {
    isValid: boolean;
    issues: string[];
  } {
    const issues: string[] = [];

    if (!data || data.length === 0) {
      issues.push('No data provided');
      return { isValid: false, issues };
    }

    // Check for invalid OHLC relationships
    data.forEach((candle, i) => {
      if (candle.high < candle.low) {
        issues.push(`Invalid candle at index ${i}: high < low`);
      }
      if (candle.high < Math.max(candle.open, candle.close)) {
        issues.push(`Invalid candle at index ${i}: high < max(open, close)`);
      }
      if (candle.low > Math.min(candle.open, candle.close)) {
        issues.push(`Invalid candle at index ${i}: low > min(open, close)`);
      }
    });

    // Check for time ordering
    for (let i = 1; i < data.length; i++) {
      const prevTime = this.getTimestamp(data[i - 1].time);
      const currTime = this.getTimestamp(data[i].time);
      if (currTime <= prevTime) {
        issues.push(`Time ordering issue at index ${i}`);
      }
    }

    return {
      isValid: issues.length === 0,
      issues,
    };
  }
}

// Singleton instance
export const marketDataNormalizer = new MarketDataNormalizer();
