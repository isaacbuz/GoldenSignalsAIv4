/**
 * useChartScales Hook
 *
 * Calculates and manages coordinate transformation scales for the chart.
 * Converts data coordinates (time/price) to canvas coordinates (x/y pixels).
 *
 * This hook handles:
 * - Price range calculation with padding
 * - X-axis scaling for time series data
 * - Y-axis scaling for price data
 * - Volume scaling for volume bars
 * - Dynamic range adjustments
 */

import { useMemo } from 'react';
import { ChartDataPoint, ChartPadding, PriceRange } from '../types';

interface UseChartScalesParams {
  /**
   * Chart data points
   */
  data: ChartDataPoint[];

  /**
   * Chart width in pixels (excluding padding)
   */
  width: number;

  /**
   * Chart height in pixels (excluding padding)
   */
  height: number;

  /**
   * Padding around the chart
   */
  padding: ChartPadding;

  /**
   * Whether to include volume in scale calculations
   */
  includeVolume?: boolean;

  /**
   * Custom price range override
   */
  customPriceRange?: { min: number; max: number };
}

interface UseChartScalesResult {
  /**
   * Scale function for X-axis (time to pixels)
   */
  xScale: (time: number, index: number) => number;

  /**
   * Scale function for Y-axis (price to pixels)
   */
  yScale: (price: number) => number;

  /**
   * Scale function for volume bars
   */
  volumeScale: (volume: number) => number;

  /**
   * Calculated price range
   */
  priceRange: PriceRange;

  /**
   * Reverse scale functions for interactions
   */
  reverseScales: {
    x: (pixelX: number) => { time: number; index: number };
    y: (pixelY: number) => number;
  };
}

/**
 * Calculate price range from data with padding
 */
const calculatePriceRange = (
  data: ChartDataPoint[],
  customRange?: { min: number; max: number }
): PriceRange => {
  // Handle undefined or null data
  if (!data || data.length === 0) {
    return {
      min: 0,
      max: 100,
      range: 100,
      paddedMin: 0,
      paddedMax: 100,
    };
  }

  // Use custom range if provided
  if (customRange) {
    const range = customRange.max - customRange.min;
    const padding = range * 0.1; // 10% padding
    return {
      min: customRange.min,
      max: customRange.max,
      range,
      paddedMin: customRange.min - padding,
      paddedMax: customRange.max + padding,
    };
  }

  // Calculate from data
  let minPrice = Infinity;
  let maxPrice = -Infinity;

  data.forEach((point) => {
    minPrice = Math.min(minPrice, point.low);
    maxPrice = Math.max(maxPrice, point.high);
  });

  const range = maxPrice - minPrice;
  const padding = range * 0.1; // Add 10% padding

  return {
    min: minPrice,
    max: maxPrice,
    range,
    paddedMin: minPrice - padding,
    paddedMax: maxPrice + padding,
  };
};

/**
 * Calculate maximum volume for scaling
 */
const calculateMaxVolume = (data: ChartDataPoint[]): number => {
  if (data.length === 0) return 1;

  return Math.max(...data.map(d => d.volume));
};

export const useChartScales = ({
  data,
  width,
  height,
  padding,
  includeVolume = true,
  customPriceRange,
}: UseChartScalesParams): UseChartScalesResult => {
  // Calculate price range
  const priceRange = useMemo(
    () => calculatePriceRange(data, customPriceRange),
    [data, customPriceRange]
  );

  // Calculate max volume
  const maxVolume = useMemo(
    () => includeVolume ? calculateMaxVolume(data) : 1,
    [data, includeVolume]
  );

  // X-axis scale function with compression for sparse data
  const xScale = useMemo(() => {
    const dataLength = Math.max(data.length, 1);

    // Calculate data density and appropriate compression
    let compressionFactor = 1;
    let candleWidth = width / dataLength;

    if (data.length > 1) {
      const timeRange = data[data.length - 1].time - data[0].time;
      const days = timeRange / (24 * 60 * 60); // Convert seconds to days
      const density = data.length / Math.max(days, 1);

      // Determine compression based on density and timeframe
      if (days > 365 * 5) {
        // 5+ years: Max compression, show compact view
        compressionFactor = Math.min(0.3, data.length / 1000);
        candleWidth = Math.max(2, width * compressionFactor / dataLength); // Min 2px per candle
      } else if (days > 365) {
        // 1-5 years: Moderate compression
        compressionFactor = Math.min(0.6, density / 2);
        candleWidth = Math.max(3, width * compressionFactor / dataLength); // Min 3px per candle
      } else if (density < 5) {
        // Less than 5 points per day: Light compression
        compressionFactor = Math.min(0.8, density / 5);
        candleWidth = Math.max(5, width * compressionFactor / dataLength); // Min 5px per candle
      } else {
        // Dense data: No compression needed
        compressionFactor = 1;
        candleWidth = width / dataLength;
      }
    }

    const effectiveWidth = candleWidth * dataLength;

    // Center compressed chart
    const xOffset = (width - effectiveWidth) / 2;

    return (time: number, index: number): number => {
      // Use index for positioning to ensure even spacing
      return padding.left + xOffset + (index + 0.5) * candleWidth;
    };
  }, [data, width, padding.left]);

  // Y-axis scale function
  const yScale = useMemo(() => {
    const { paddedMin, paddedMax } = priceRange;
    const priceRangeSize = paddedMax - paddedMin || 1; // Prevent division by zero

    return (price: number): number => {
      // Normalize price to 0-1 range
      const normalized = (price - paddedMin) / priceRangeSize;

      // Invert and scale to canvas coordinates (top = high price)
      return padding.top + (1 - normalized) * height;
    };
  }, [priceRange, height, padding.top]);

  // Volume scale function
  const volumeScale = useMemo(() => {
    const volumeHeight = height * 0.2; // Volume takes 20% of chart height

    return (volume: number): number => {
      if (maxVolume === 0) return 0;

      // Normalize volume to 0-1 range
      const normalized = volume / maxVolume;

      // Scale to volume height
      return normalized * volumeHeight;
    };
  }, [height, maxVolume]);

  // Reverse scale functions for mouse interactions
  const reverseScales = useMemo(() => {
    const dataLength = Math.max(data.length, 1);
    const candleWidth = width / dataLength;
    const { paddedMin, paddedMax } = priceRange;
    const priceRangeSize = paddedMax - paddedMin || 1;

    return {
      // Convert pixel X to time and index
      x: (pixelX: number): { time: number; index: number } => {
        const relativeX = pixelX - padding.left;
        const index = Math.floor(relativeX / candleWidth);
        const clampedIndex = Math.max(0, Math.min(index, data.length - 1));

        return {
          time: data[clampedIndex]?.time || 0,
          index: clampedIndex,
        };
      },

      // Convert pixel Y to price
      y: (pixelY: number): number => {
        const relativeY = pixelY - padding.top;
        const normalized = 1 - (relativeY / height);

        return paddedMin + normalized * priceRangeSize;
      },
    };
  }, [data, width, height, padding, priceRange]);

  return {
    xScale,
    yScale,
    volumeScale,
    priceRange,
    reverseScales,
  };
};
