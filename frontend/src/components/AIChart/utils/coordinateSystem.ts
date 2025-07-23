/**
 * Coordinate System Utility
 *
 * Handles all coordinate transformations between data space and canvas space.
 * Provides reliable mapping for time/price to pixel coordinates and vice versa.
 */

export interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface PriceRange {
  min: number;
  max: number;
}

export interface TimeRange {
  start: number;
  end: number;
}

export interface ViewportBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export class CoordinateSystem {
  private viewportBounds: ViewportBounds;
  private logScale: boolean = false;
  private indexMapping?: { time: number; index: number }[];

  constructor(
    private canvasWidth: number,
    private canvasHeight: number,
    private padding: Padding,
    private priceRange: PriceRange,
    private timeRange: TimeRange
  ) {
    // Calculate viewport bounds (area inside padding)
    this.viewportBounds = {
      x: padding.left,
      y: padding.top,
      width: canvasWidth - padding.left - padding.right,
      height: canvasHeight - padding.top - padding.bottom
    };
  }

  /**
   * Enable/disable logarithmic price scale
   */
  setLogScale(enabled: boolean): void {
    this.logScale = enabled;
  }

  /**
   * Set index mapping for evenly spaced candles
   */
  setIndexMapping(data: { time: number }[]): void {
    this.indexMapping = data.map((d, i) => ({ time: d.time, index: i }));
  }

  /**
   * Convert timestamp to X coordinate
   */
  timeToX(timestamp: number): number {
    // If index mapping is available, use it for even spacing
    if (this.indexMapping && this.indexMapping.length > 0) {
      // Find the closest data point
      const dataPoint = this.indexMapping.find(d => d.time === timestamp);
      if (dataPoint) {
        // Use index-based positioning for even spacing
        const totalPoints = this.indexMapping.length;
        const ratio = dataPoint.index / (totalPoints - 1 || 1);
        return this.viewportBounds.x + ratio * this.viewportBounds.width;
      }
    }

    // Fallback to time-based positioning
    const ratio = (timestamp - this.timeRange.start) /
                  (this.timeRange.end - this.timeRange.start);
    return this.viewportBounds.x + ratio * this.viewportBounds.width;
  }

  /**
   * Convert price to Y coordinate
   */
  priceToY(price: number): number {
    let ratio: number;

    if (this.logScale && this.priceRange.min > 0) {
      // Logarithmic scale
      const logMin = Math.log(this.priceRange.min);
      const logMax = Math.log(this.priceRange.max);
      const logPrice = Math.log(price);
      ratio = (logPrice - logMin) / (logMax - logMin);
    } else {
      // Linear scale
      ratio = (price - this.priceRange.min) /
              (this.priceRange.max - this.priceRange.min);
    }

    // Invert Y axis (canvas Y increases downward)
    return this.viewportBounds.y + this.viewportBounds.height -
           (ratio * this.viewportBounds.height);
  }

  /**
   * Convert X coordinate to timestamp
   */
  xToTime(x: number): number {
    const ratio = (x - this.viewportBounds.x) / this.viewportBounds.width;
    return this.timeRange.start + ratio * (this.timeRange.end - this.timeRange.start);
  }

  /**
   * Convert Y coordinate to price
   */
  yToPrice(y: number): number {
    // Invert Y axis calculation
    const ratio = (this.viewportBounds.y + this.viewportBounds.height - y) /
                  this.viewportBounds.height;

    if (this.logScale && this.priceRange.min > 0) {
      // Logarithmic scale
      const logMin = Math.log(this.priceRange.min);
      const logMax = Math.log(this.priceRange.max);
      const logPrice = logMin + ratio * (logMax - logMin);
      return Math.exp(logPrice);
    } else {
      // Linear scale
      return this.priceRange.min + ratio * (this.priceRange.max - this.priceRange.min);
    }
  }

  /**
   * Get the pixel distance for a given price difference
   */
  priceToPixelDistance(priceDiff: number, atPrice: number): number {
    const y1 = this.priceToY(atPrice);
    const y2 = this.priceToY(atPrice + priceDiff);
    return Math.abs(y2 - y1);
  }

  /**
   * Get the pixel distance for a given time difference
   */
  timeToPixelDistance(timeDiff: number): number {
    const x1 = this.timeToX(0);
    const x2 = this.timeToX(timeDiff);
    return Math.abs(x2 - x1);
  }

  /**
   * Check if a point is within the viewport
   */
  isInViewport(x: number, y: number): boolean {
    return x >= this.viewportBounds.x &&
           x <= this.viewportBounds.x + this.viewportBounds.width &&
           y >= this.viewportBounds.y &&
           y <= this.viewportBounds.y + this.viewportBounds.height;
  }

  /**
   * Get nice round numbers for price grid lines
   */
  getNicePriceIntervals(targetCount: number = 10): number[] {
    const range = this.priceRange.max - this.priceRange.min;
    const roughInterval = range / targetCount;

    // Find nice round interval
    const magnitude = Math.pow(10, Math.floor(Math.log10(roughInterval)));
    const normalized = roughInterval / magnitude;

    let niceInterval: number;
    if (normalized <= 1) niceInterval = magnitude;
    else if (normalized <= 2) niceInterval = 2 * magnitude;
    else if (normalized <= 5) niceInterval = 5 * magnitude;
    else niceInterval = 10 * magnitude;

    // Generate intervals
    const intervals: number[] = [];
    const start = Math.ceil(this.priceRange.min / niceInterval) * niceInterval;

    for (let price = start; price <= this.priceRange.max; price += niceInterval) {
      intervals.push(price);
    }

    return intervals;
  }

  /**
   * Get nice time intervals for time grid lines
   */
  getNiceTimeIntervals(targetCount: number = 10): number[] {
    const range = this.timeRange.end - this.timeRange.start;
    const roughInterval = range / targetCount;

    // Define nice time intervals in milliseconds
    const niceIntervals = [
      1000,           // 1 second
      5000,           // 5 seconds
      10000,          // 10 seconds
      30000,          // 30 seconds
      60000,          // 1 minute
      300000,         // 5 minutes
      900000,         // 15 minutes
      1800000,        // 30 minutes
      3600000,        // 1 hour
      7200000,        // 2 hours
      14400000,       // 4 hours
      86400000,       // 1 day
      604800000,      // 1 week
      2592000000,     // 30 days
    ];

    // Find closest nice interval
    let niceInterval = niceIntervals[0];
    for (const interval of niceIntervals) {
      if (interval >= roughInterval) {
        niceInterval = interval;
        break;
      }
    }

    // Generate intervals
    const intervals: number[] = [];
    const start = Math.ceil(this.timeRange.start / niceInterval) * niceInterval;

    for (let time = start; time <= this.timeRange.end; time += niceInterval) {
      intervals.push(time);
    }

    return intervals;
  }

  /**
   * Update ranges (for zoom/pan operations)
   */
  updateRanges(priceRange?: PriceRange, timeRange?: TimeRange): void {
    if (priceRange) {
      this.priceRange = priceRange;
    }
    if (timeRange) {
      this.timeRange = timeRange;
    }
  }

  /**
   * Get viewport bounds
   */
  getViewportBounds(): ViewportBounds {
    return { ...this.viewportBounds };
  }

  /**
   * Get current ranges
   */
  getRanges(): { price: PriceRange; time: TimeRange } {
    return {
      price: { ...this.priceRange },
      time: { ...this.timeRange }
    };
  }

  /**
   * Clone the coordinate system
   */
  clone(): CoordinateSystem {
    const clone = new CoordinateSystem(
      this.canvasWidth,
      this.canvasHeight,
      { ...this.padding },
      { ...this.priceRange },
      { ...this.timeRange }
    );
    clone.setLogScale(this.logScale);
    return clone;
  }
}
