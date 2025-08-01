/**
 * ChartSignalAgent - Manages chart signal visualization and positioning
 *
 * This agent is responsible for:
 * 1. Receiving signals from the backend/consensus
 * 2. Converting price/time coordinates to chart pixel positions
 * 3. Managing signal overlays and animations
 * 4. Updating positions on chart interactions
 */

import { IChartApi, Time } from 'lightweight-charts';
import { Subject, BehaviorSubject, fromEvent, merge } from 'rxjs';
import { debounceTime, filter } from 'rxjs/operators';
import logger from '../services/logger';


export interface ChartSignal {
  id: string;
  type: 'buy' | 'sell';
  price: number;
  time: Time;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number[];
  reasoning?: string;
  source?: string; // Which agent generated this signal
  consensus?: {
    agentsInFavor: number;
    totalAgents: number;
    confidence: number;
  };
}

export interface SignalOverlay extends ChartSignal {
  coordinates?: {
    x: number;
    y: number;
  };
  visible: boolean;
  opacity: number;
}

export interface ChartSignalAgentConfig {
  chart: IChartApi | null;
  updateInterval?: number;
  showConfidenceThreshold?: number;
  enableRealTimeUpdates?: boolean;
}

export class ChartSignalAgent {
  private chart: IChartApi | null = null;
  private signals = new BehaviorSubject<SignalOverlay[]>([]);
  private coordinateUpdateSubject = new Subject<void>();
  private config: ChartSignalAgentConfig;
  private wsConnection: WebSocket | null = null;

  constructor(config: ChartSignalAgentConfig) {
    this.config = {
      updateInterval: 100,
      showConfidenceThreshold: 0.7,
      enableRealTimeUpdates: true,
      ...config,
    };

    this.chart = config.chart;
    this.setupCoordinateUpdates();
  }

  /**
   * Initialize the agent with a chart instance
   */
  public setChart(chart: IChartApi): void {
    this.chart = chart;
    this.setupChartListeners();
    this.updateAllCoordinates();
  }

  /**
   * Connect to backend WebSocket for real-time signals
   */
  public connectToBackend(wsUrl: string): void {
    if (!this.config.enableRealTimeUpdates) return;

    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'signal') {
        this.addSignal(data.signal);
      } else if (data.type === 'consensus') {
        this.updateSignalsFromConsensus(data.consensus);
      }
    };

    this.wsConnection.onerror = (error) => {
      logger.error('ChartSignalAgent WebSocket error:', error);
    };
  }

  /**
   * Add a new signal to the chart
   */
  public addSignal(signal: ChartSignal): void {
    if (signal.confidence < (this.config.showConfidenceThreshold || 0.7)) {
      return; // Don't show low confidence signals
    }

    const overlay: SignalOverlay = {
      ...signal,
      visible: true,
      opacity: this.calculateOpacity(signal.confidence),
      coordinates: undefined, // Will be calculated by updateCoordinates
    };

    const currentSignals = this.signals.value;

    // Remove duplicate signals at the same price level
    const filtered = currentSignals.filter(s =>
      !(Math.abs(s.price - signal.price) < 0.01 && s.type === signal.type)
    );

    this.signals.next([...filtered, overlay]);

    // Immediately try to calculate coordinates if chart is available
    if (this.chart) {
      setTimeout(() => this.updateCoordinates([overlay]), 100);
    }
  }

  /**
   * Update signals based on multi-agent consensus
   */
  public updateSignalsFromConsensus(consensus: any): void {
    // Extract high-confidence signals from consensus
    const signals: ChartSignal[] = (consensus.signals || [])
      .filter((s: any) => s.confidence >= (this.config?.showConfidenceThreshold || 0.7))
      .map((s: any) => ({
        id: `consensus-${Date.now()}-${Math.random()}`,
        type: s.action.toLowerCase() as 'buy' | 'sell',
        price: s.price,
        time: s.time || (Date.now() / 1000) as Time,
        confidence: s.confidence,
        stopLoss: s.stopLoss,
        takeProfit: s.takeProfit,
        reasoning: s.reasoning,
        consensus: {
          agentsInFavor: s.agentsInFavor || consensus.agentsInFavor,
          totalAgents: consensus.totalAgents || 30,
          confidence: consensus.confidence,
        },
      }));

    signals.forEach(signal => this.addSignal(signal));
  }

  /**
   * Get current signals as observable
   */
  public getSignals() {
    return this.signals.asObservable();
  }

  /**
   * Calculate opacity based on confidence
   */
  private calculateOpacity(confidence: number): number {
    // Higher confidence = more opaque
    return 0.5 + (confidence * 0.5);
  }

  /**
   * Setup coordinate update system
   */
  private setupCoordinateUpdates(): void {
    this.coordinateUpdateSubject
      .pipe(debounceTime(this.config.updateInterval || 100))
      .subscribe(() => {
        this.updateAllCoordinates();
      });
  }

  /**
   * Setup chart event listeners
   */
  private setupChartListeners(): void {
    if (!this.chart) return;

    // Update on visible range change
    this.chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      this.coordinateUpdateSubject.next();
    });

    // Update on crosshair move (for hover effects)
    this.chart.subscribeCrosshairMove((param) => {
      if (param.point) {
        this.updateHoverStates(param.point);
      }
    });
  }

  /**
   * Update coordinates for specific signals
   */
  private updateCoordinates(signals: SignalOverlay[]): void {
    if (!this.chart) return;

    signals.forEach(signal => {
      try {
        const x = this.chart!.timeScale().timeToCoordinate(signal.time);
        // Use coordinateToPrice as a workaround for missing priceToCoordinate
        const priceScale = this.chart!.priceScale('right');
        const y = (priceScale as any).priceToCoordinate ? 
          (priceScale as any).priceToCoordinate(signal.price) : null;

        if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
          signal.coordinates = { x, y };
          logger.info(`Signal ${signal.id} positioned at (${x}, ${y}) for price ${signal.price}`);
        } else {
          logger.warn(`Failed to calculate coordinates for signal ${signal.id}`, { x, y, price: signal.price, time: signal.time });
        }
      } catch (error) {
        logger.error(`Error calculating coordinates for signal ${signal.id}:`, error);
      }
    });

    this.signals.next([...this.signals.value]);
  }

  /**
   * Update all signal coordinates
   */
  private updateAllCoordinates(): void {
    if (!this.chart) return;

    const updatedSignals = this.signals.value.map(signal => {
      try {
        const x = this.chart!.timeScale().timeToCoordinate(signal.time);
        // Use coordinateToPrice as a workaround for missing priceToCoordinate
        const priceScale = this.chart!.priceScale('right');
        const y = (priceScale as any).priceToCoordinate ? 
          (priceScale as any).priceToCoordinate(signal.price) : null;

        if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
          return {
            ...signal,
            coordinates: { x, y },
            visible: true, // Signal is in visible range
          };
        } else {
          return {
            ...signal,
            visible: false, // Signal is outside visible range
          };
        }
      } catch (error) {
        logger.error(`Error updating coordinates for signal ${signal.id}:`, error);
        return {
          ...signal,
          visible: false,
        };
      }
    });

    this.signals.next(updatedSignals);
  }

  /**
   * Update hover states based on crosshair position
   */
  private updateHoverStates(point: { x: number; y: number }): void {
    const updatedSignals = this.signals.value.map(signal => {
      if (!signal.coordinates || !signal.visible) return signal;

      const distance = Math.sqrt(
        Math.pow(signal.coordinates.x - point.x, 2) +
        Math.pow(signal.coordinates.y - point.y, 2)
      );

      // Highlight signals near the cursor
      const isHovered = distance < 50;

      return {
        ...signal,
        opacity: isHovered ? 1 : this.calculateOpacity(signal.confidence),
      };
    });

    this.signals.next(updatedSignals);
  }

  /**
   * Remove a signal
   */
  public removeSignal(signalId: string): void {
    const filtered = this.signals.value.filter(s => s.id !== signalId);
    this.signals.next(filtered);
  }

  /**
   * Clear all signals
   */
  public clearSignals(): void {
    this.signals.next([]);
  }

  /**
   * Get signals within a price range
   */
  public getSignalsInRange(minPrice: number, maxPrice: number): SignalOverlay[] {
    return this.signals.value.filter(s =>
      s.price >= minPrice && s.price <= maxPrice && s.visible
    );
  }

  /**
   * Analyze chart patterns and generate signals
   * This method can be called periodically or on-demand
   */
  public async analyzeChart(): Promise<ChartSignal[]> {
    if (!this.chart) return [];

    // This would connect to backend analysis
    // For now, return empty array
    return [];
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    try {
      if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
        this.wsConnection.close();
      }
    } catch (error) {
      // Silently handle WebSocket close errors
    }
    this.signals.complete();
    this.coordinateUpdateSubject.complete();
  }
}

/**
 * Factory function to create and initialize ChartSignalAgent
 */
export function createChartSignalAgent(chart: IChartApi | null): ChartSignalAgent {
  return new ChartSignalAgent({
    chart,
    updateInterval: 50, // Fast updates for smooth movement
    showConfidenceThreshold: 0.75, // Only show high confidence signals
    enableRealTimeUpdates: true,
  });
}
