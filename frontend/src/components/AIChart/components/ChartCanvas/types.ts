/**
 * ChartCanvas Types
 *
 * Type definitions for the ChartCanvas component and its related utilities
 */

export interface ChartDataPoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  isGapFilled?: boolean;
}

export interface PriceRange {
  min: number;
  max: number;
}

export interface TimeRange {
  start: number;
  end: number;
}

export interface DrawingContext {
  ctx: CanvasRenderingContext2D;
  data: ChartDataPoint[];
  coordinates: any; // Will use CoordinateSystem
  theme: any;
}

export interface AgentSignal {
  id: string;
  agentName: string;
  type: 'buy' | 'sell' | 'hold';
  price: number;
  time: number;
  confidence: number;
  reason?: string;
}

export interface TradingLevel {
  type: 'support' | 'resistance' | 'entry' | 'stop' | 'target';
  price: number;
  strength?: number;
  label?: string;
}

export interface ChartPattern {
  type: string;
  name: string;
  points: Array<{ time: number; price: number }>;
  confidence: number;
  targetPrice?: number;
}
