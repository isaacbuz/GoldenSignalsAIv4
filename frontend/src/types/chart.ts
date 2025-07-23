import { Time } from 'lightweight-charts';

export interface ChartDataPoint {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionPoint {
  time: number;
  price: number;
  upperBound?: number;
  lowerBound?: number;
  confidence?: number;
}

export interface Pattern {
  id: string;
  type: string;
  points: Array<{
    time: number;
    price: number;
  }>;
  confidence: number;
  startTime?: number;
  endTime?: number;
}

export interface Signal {
  id: string;
  symbol: string;
  time: number;
  price: number;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning?: string;
  target_price?: number;
  stop_loss?: number;
  take_profits?: number[];
  risk_reward_ratio?: number;
  agents_consensus?: {
    agentsInFavor: number;
    totalAgents: number;
  };
}

export interface Indicator {
  name: string;
  value: number;
  signal?: string;
  data?: number[];
}

export interface AIAnalysisResult {
  predictions: PredictionPoint[];
  patterns: Pattern[];
  indicators: Indicator[];
  currentPrice: number;
  supportLevel: number;
  resistanceLevel: number;
  confidence: number;
  reasoning: string;
}

export interface TradingAdvice {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  entry_price: number;
  stop_loss: number;
  take_profits: number[];
  confidence: number;
  reasoning: string;
  risk_reward_ratio: number;
  expires_at?: number;
}

export interface MarketData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface AgentConsensus {
  time: number;
  price: number;
  agreement: number; // 0-1 percentage
  votes: Array<{
    agentId: string;
    signal: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
  }>;
}
