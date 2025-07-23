/**
 * Chart Controller Interface
 *
 * Provides programmatic control over the AITradingChart component.
 * This controller enables Golden Eye and other systems to manipulate
 * the chart through a clean API interface.
 */

import { EventEmitter } from 'events';

// Types
export interface PredictionParams {
  symbol: string;
  points: Array<{
    time: number;
    price: number;
  }>;
  confidence?: {
    upper: Array<{ time: number; price: number }>;
    lower: Array<{ time: number; price: number }>;
  };
  horizon: number; // hours
  metadata?: {
    model: string;
    confidence: number;
    factors: string[];
  };
}

export interface AgentSignal {
  id: string;
  agentName: string;
  time: number;
  price: number;
  type: 'buy' | 'sell' | 'hold';
  confidence: number;
  reasoning?: string;
}

export interface PriceLevel {
  price: number;
  type: 'support' | 'resistance';
  strength: number; // 0-1
  touches: number;
  label?: string;
}

export interface TradingZone {
  startPrice: number;
  endPrice: number;
  type: 'entry' | 'exit' | 'stopLoss' | 'takeProfit';
  label: string;
  color?: string;
  opacity?: number;
}

export interface ChartPattern {
  id: string;
  name: string;
  type: 'triangle' | 'head_shoulders' | 'double_top' | 'double_bottom' | 'flag' | 'wedge';
  startTime: number;
  endTime: number;
  keyPoints: Array<{ time: number; price: number }>;
  confidence: number;
}

export interface ChartAnnotation {
  id: string;
  type: 'text' | 'arrow' | 'circle' | 'rectangle' | 'line';
  data: any; // Type-specific data
  style?: any;
  persistent?: boolean;
}

export interface ChartSnapshot {
  imageData: string; // base64
  timestamp: number;
  annotations: ChartAnnotation[];
  activeIndicators: string[];
}

/**
 * Chart Controller Interface
 * All chart manipulation methods
 */
export interface IChartController {
  // Prediction methods
  drawPrediction(params: PredictionParams): Promise<void>;
  updatePrediction(id: string, params: Partial<PredictionParams>): void;
  clearPrediction(id?: string): void;

  // Signal methods
  addAgentSignal(signal: AgentSignal): void;
  updateSignal(id: string, updates: Partial<AgentSignal>): void;
  removeSignal(id: string): void;
  highlightSignals(agentName: string): void;
  clearSignals(agentName?: string): void;

  // Level methods
  drawSupportResistance(levels: PriceLevel[]): void;
  updateLevel(price: number, updates: Partial<PriceLevel>): void;
  clearLevels(type?: 'support' | 'resistance'): void;
  drawEntryExitZones(zones: TradingZone[]): void;
  clearZones(type?: string): void;

  // Pattern methods
  highlightPattern(pattern: ChartPattern): void;
  updatePattern(id: string, updates: Partial<ChartPattern>): void;
  clearPatterns(type?: string): void;
  animatePattern(id: string, animation: 'pulse' | 'flash' | 'glow'): void;

  // Annotation methods
  addAnnotation(annotation: ChartAnnotation): string; // returns id
  updateAnnotation(id: string, updates: Partial<ChartAnnotation>): void;
  removeAnnotation(id: string): void;
  clearAnnotations(type?: string): void;

  // View control
  zoomToTimeRange(startTime: number, endTime: number, animate?: boolean): void;
  zoomToFit(padding?: number): void;
  panToPrice(price: number, animate?: boolean): void;
  panToTime(time: number, animate?: boolean): void;
  resetView(): void;

  // Data methods
  getCurrentPrice(): number;
  getVisibleRange(): { startTime: number; endTime: number; minPrice: number; maxPrice: number };
  getDataAtPoint(x: number, y: number): { time: number; price: number; candle?: any };

  // Screenshot/Export
  getSnapshot(options?: { width?: number; height?: number; format?: string }): Promise<ChartSnapshot>;
  exportData(format: 'csv' | 'json'): string;

  // Configuration
  setIndicators(indicators: string[]): void;
  setTimeframe(timeframe: string): void;
  setChartType(type: 'candle' | 'line' | 'bar'): void;
  setTheme(theme: 'light' | 'dark'): void;

  // Event handling
  on(event: string, handler: Function): void;
  off(event: string, handler: Function): void;

  // State
  getState(): any;
  setState(state: any): void;
}

/**
 * Base implementation of Chart Controller
 */
export class ChartController extends EventEmitter implements IChartController {
  private chartRef: any; // Reference to the actual chart component
  private predictions: Map<string, PredictionParams> = new Map();
  private signals: Map<string, AgentSignal> = new Map();
  private patterns: Map<string, ChartPattern> = new Map();
  private annotations: Map<string, ChartAnnotation> = new Map();
  private levels: PriceLevel[] = [];
  private zones: TradingZone[] = [];

  constructor(chartRef: any) {
    super();
    this.chartRef = chartRef;
  }

  // Prediction methods
  async drawPrediction(params: PredictionParams): Promise<void> {
    const id = `prediction_${Date.now()}`;
    this.predictions.set(id, params);

    // Emit event for chart to handle
    this.emit('drawPrediction', { id, ...params });

    // Animate the drawing
    await this.animatePredictionDraw(id, params);
  }

  updatePrediction(id: string, params: Partial<PredictionParams>): void {
    const existing = this.predictions.get(id);
    if (existing) {
      this.predictions.set(id, { ...existing, ...params });
      this.emit('updatePrediction', { id, ...params });
    }
  }

  clearPrediction(id?: string): void {
    if (id) {
      this.predictions.delete(id);
      this.emit('clearPrediction', { id });
    } else {
      this.predictions.clear();
      this.emit('clearAllPredictions');
    }
  }

  // Signal methods
  addAgentSignal(signal: AgentSignal): void {
    this.signals.set(signal.id, signal);
    this.emit('addSignal', signal);

    // Animate signal appearance
    this.animateSignalEntry(signal);
  }

  updateSignal(id: string, updates: Partial<AgentSignal>): void {
    const existing = this.signals.get(id);
    if (existing) {
      this.signals.set(id, { ...existing, ...updates });
      this.emit('updateSignal', { id, ...updates });
    }
  }

  removeSignal(id: string): void {
    this.signals.delete(id);
    this.emit('removeSignal', { id });
  }

  highlightSignals(agentName: string): void {
    const agentSignals = Array.from(this.signals.values())
      .filter(s => s.agentName === agentName);
    this.emit('highlightSignals', { agentName, signals: agentSignals });
  }

  clearSignals(agentName?: string): void {
    if (agentName) {
      Array.from(this.signals.entries())
        .filter(([_, signal]) => signal.agentName === agentName)
        .forEach(([id]) => this.signals.delete(id));
      this.emit('clearSignals', { agentName });
    } else {
      this.signals.clear();
      this.emit('clearAllSignals');
    }
  }

  // Level methods
  drawSupportResistance(levels: PriceLevel[]): void {
    this.levels = levels;
    this.emit('drawLevels', { levels });
  }

  updateLevel(price: number, updates: Partial<PriceLevel>): void {
    const index = this.levels.findIndex(l => Math.abs(l.price - price) < 0.01);
    if (index !== -1) {
      this.levels[index] = { ...this.levels[index], ...updates };
      this.emit('updateLevel', { price, updates });
    }
  }

  clearLevels(type?: 'support' | 'resistance'): void {
    if (type) {
      this.levels = this.levels.filter(l => l.type !== type);
      this.emit('clearLevels', { type });
    } else {
      this.levels = [];
      this.emit('clearAllLevels');
    }
  }

  drawEntryExitZones(zones: TradingZone[]): void {
    this.zones = zones;
    this.emit('drawZones', { zones });
  }

  clearZones(type?: string): void {
    if (type) {
      this.zones = this.zones.filter(z => z.type !== type);
      this.emit('clearZones', { type });
    } else {
      this.zones = [];
      this.emit('clearAllZones');
    }
  }

  // Pattern methods
  highlightPattern(pattern: ChartPattern): void {
    this.patterns.set(pattern.id, pattern);
    this.emit('highlightPattern', pattern);

    // Start pattern animation
    this.animatePattern(pattern.id, 'pulse');
  }

  updatePattern(id: string, updates: Partial<ChartPattern>): void {
    const existing = this.patterns.get(id);
    if (existing) {
      this.patterns.set(id, { ...existing, ...updates });
      this.emit('updatePattern', { id, ...updates });
    }
  }

  clearPatterns(type?: string): void {
    if (type) {
      Array.from(this.patterns.entries())
        .filter(([_, pattern]) => pattern.type === type)
        .forEach(([id]) => this.patterns.delete(id));
      this.emit('clearPatterns', { type });
    } else {
      this.patterns.clear();
      this.emit('clearAllPatterns');
    }
  }

  animatePattern(id: string, animation: 'pulse' | 'flash' | 'glow'): void {
    this.emit('animatePattern', { id, animation });
  }

  // Annotation methods
  addAnnotation(annotation: ChartAnnotation): string {
    const id = annotation.id || `annotation_${Date.now()}`;
    this.annotations.set(id, { ...annotation, id });
    this.emit('addAnnotation', { ...annotation, id });
    return id;
  }

  updateAnnotation(id: string, updates: Partial<ChartAnnotation>): void {
    const existing = this.annotations.get(id);
    if (existing) {
      this.annotations.set(id, { ...existing, ...updates });
      this.emit('updateAnnotation', { id, ...updates });
    }
  }

  removeAnnotation(id: string): void {
    this.annotations.delete(id);
    this.emit('removeAnnotation', { id });
  }

  clearAnnotations(type?: string): void {
    if (type) {
      Array.from(this.annotations.entries())
        .filter(([_, annotation]) => annotation.type === type)
        .forEach(([id]) => this.annotations.delete(id));
      this.emit('clearAnnotations', { type });
    } else {
      this.annotations.clear();
      this.emit('clearAllAnnotations');
    }
  }

  // View control
  zoomToTimeRange(startTime: number, endTime: number, animate = true): void {
    this.emit('zoomToTimeRange', { startTime, endTime, animate });
  }

  zoomToFit(padding = 0.1): void {
    this.emit('zoomToFit', { padding });
  }

  panToPrice(price: number, animate = true): void {
    this.emit('panToPrice', { price, animate });
  }

  panToTime(time: number, animate = true): void {
    this.emit('panToTime', { time, animate });
  }

  resetView(): void {
    this.emit('resetView');
  }

  // Data methods
  getCurrentPrice(): number {
    // This would be implemented by the chart component
    return this.chartRef?.getCurrentPrice() || 0;
  }

  getVisibleRange(): { startTime: number; endTime: number; minPrice: number; maxPrice: number } {
    return this.chartRef?.getVisibleRange() || {
      startTime: 0,
      endTime: 0,
      minPrice: 0,
      maxPrice: 0
    };
  }

  getDataAtPoint(x: number, y: number): { time: number; price: number; candle?: any } {
    return this.chartRef?.getDataAtPoint(x, y) || {
      time: 0,
      price: 0
    };
  }

  // Screenshot/Export
  async getSnapshot(options?: { width?: number; height?: number; format?: string }): Promise<ChartSnapshot> {
    const canvas = this.chartRef?.getCanvas();
    if (!canvas) {
      throw new Error('Chart canvas not available');
    }

    const imageData = canvas.toDataURL(options?.format || 'image/png');

    return {
      imageData,
      timestamp: Date.now(),
      annotations: Array.from(this.annotations.values()),
      activeIndicators: this.chartRef?.getIndicators() || []
    };
  }

  exportData(format: 'csv' | 'json'): string {
    const data = this.chartRef?.getData() || [];

    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else {
      // CSV export
      const headers = ['time', 'open', 'high', 'low', 'close', 'volume'];
      const rows = data.map((d: any) =>
        [d.time, d.open, d.high, d.low, d.close, d.volume].join(',')
      );
      return [headers.join(','), ...rows].join('\n');
    }
  }

  // Configuration
  setIndicators(indicators: string[]): void {
    this.emit('setIndicators', { indicators });
  }

  setTimeframe(timeframe: string): void {
    this.emit('setTimeframe', { timeframe });
  }

  setChartType(type: 'candle' | 'line' | 'bar'): void {
    this.emit('setChartType', { type });
  }

  setTheme(theme: 'light' | 'dark'): void {
    this.emit('setTheme', { theme });
  }

  // State
  getState(): any {
    return {
      predictions: Array.from(this.predictions.entries()),
      signals: Array.from(this.signals.entries()),
      patterns: Array.from(this.patterns.entries()),
      annotations: Array.from(this.annotations.entries()),
      levels: this.levels,
      zones: this.zones
    };
  }

  setState(state: any): void {
    if (state.predictions) {
      this.predictions = new Map(state.predictions);
    }
    if (state.signals) {
      this.signals = new Map(state.signals);
    }
    if (state.patterns) {
      this.patterns = new Map(state.patterns);
    }
    if (state.annotations) {
      this.annotations = new Map(state.annotations);
    }
    if (state.levels) {
      this.levels = state.levels;
    }
    if (state.zones) {
      this.zones = state.zones;
    }

    this.emit('stateRestored', state);
  }

  // Private helper methods
  private async animatePredictionDraw(id: string, params: PredictionParams): Promise<void> {
    // Implement smooth animation for prediction drawing
    return new Promise(resolve => {
      let progress = 0;
      const animate = () => {
        progress += 0.05;
        this.emit('animatePrediction', { id, progress });

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          resolve();
        }
      };
      requestAnimationFrame(animate);
    });
  }

  private animateSignalEntry(signal: AgentSignal): void {
    // Implement signal entry animation
    this.emit('animateSignal', {
      id: signal.id,
      animation: 'bounceIn'
    });
  }
}

// Export factory function
export function createChartController(chartRef: any): IChartController {
  return new ChartController(chartRef);
}
