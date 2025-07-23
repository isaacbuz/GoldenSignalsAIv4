/**
 * Agent System Type Definitions
 * Comprehensive types for the multi-agent trading system
 */

// Agent signal types
export interface AgentSignal {
  agent_name: string;
  signal: 'buy' | 'sell' | 'hold';
  confidence: number;
  metadata: {
    indicator_value?: number;
    reason?: string;
    timestamp: string;
  };
}

// Workflow decision types
export interface WorkflowDecision {
  execute: boolean;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  position_size: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  max_exposure: number;
  reasoning: string;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  execution_time: string;
}

// Trading levels for chart display
export interface TradingLevels {
  entry: number;
  stopLoss: number;
  takeProfits: number[];
  action: 'BUY' | 'SELL';
  confidence: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
}

// Historical pattern matching
export interface PatternMatch {
  date: string;
  outcome: string;
  return: number;
  relevance: number;
}

export interface HistoricalInsights {
  similar_setups_found: number;
  historical_success_rate: number;
  average_return: number;
  key_patterns: PatternMatch[];
}

// Consensus and voting
export interface ConsensusResult {
  action: string;
  confidence: number;
  weighted_votes: {
    buy: number;
    sell: number;
    hold: number;
  };
  agents_total: number;
  agents_error: number;
}

// Risk assessment
export interface RiskAssessment {
  risk_score: number;
  position_size: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  risk_reward_ratio: number;
}

// Market state
export type MarketState = 'trending_up' | 'trending_down' | 'ranging' | 'volatile' | 'unknown';

// Complete workflow result
export interface WorkflowResult {
  symbol: string;
  market_state: MarketState;
  agent_signals: Record<string, AgentSignal>;
  historical_insights?: HistoricalInsights;
  consensus: ConsensusResult;
  risk_assessment: RiskAssessment;
  final_decision: WorkflowDecision;
  messages: string[];
  timestamp: string;
}

// Agent status types
export interface AgentPerformanceMetrics {
  total_signals: number;
  correct_signals: number;
  profit_factor: number;
  average_confidence: number;
  sharpe_ratio?: number;
}

export interface AgentStatus {
  name: string;
  type: string;
  status: 'active' | 'error' | 'offline';
  accuracy: number;
  avg_confidence: number;
  last_signal_time: string;
  performance_metrics: AgentPerformanceMetrics;
}

// WebSocket message types
export type WorkflowStage =
  | 'market_regime'
  | 'collecting_signals'
  | 'searching_patterns'
  | 'building_consensus'
  | 'assessing_risk'
  | 'making_decision'
  | 'complete';

export interface WorkflowUpdate {
  stage: WorkflowStage;
  progress: number; // 0-100
  message: string;
  partial_data?: Partial<WorkflowResult>;
}

export interface ErrorMessage {
  code: string;
  message: string;
  details?: any;
}

export interface AgentWebSocketMessage {
  type: 'agent_signal' | 'workflow_update' | 'agent_status' | 'error';
  data: AgentSignal | WorkflowUpdate | AgentStatus | ErrorMessage;
  timestamp: string;
}

// Service response types
export interface ServiceResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  cached?: boolean;
}

// Agent configuration
export interface AgentConfig {
  enabled: boolean;
  weight: number;
  minConfidence: number;
  parameters?: Record<string, any>;
}

export interface AgentSystemConfig {
  agents: Record<string, AgentConfig>;
  consensusThreshold: number;
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  executionMode: 'manual' | 'semi-auto' | 'auto';
}

// Analysis state for UI
export interface AnalysisState {
  status: 'idle' | 'loading' | 'success' | 'error';
  error: Error | null;
  data: WorkflowResult | null;
  progress: number;
  currentStage: WorkflowStage | null;
  messages: string[];
}

// Chart-specific signal format
export interface ChartSignal {
  agent: string;
  type: 'buy' | 'sell';
  confidence: number;
  reason: string;
  strength: number;
  timestamp?: string;
}

// Agent names enum for type safety
export enum AgentName {
  RSI = 'rsi_agent',
  MACD = 'macd_agent',
  Volume = 'volume_agent',
  Momentum = 'momentum_agent',
  Pattern = 'pattern_agent',
  Sentiment = 'sentiment_agent',
  LSTM = 'lstm_agent',
  Options = 'options_agent',
  MarketRegime = 'market_regime_agent'
}

// Agent weights configuration
export const AGENT_WEIGHTS: Record<AgentName, number> = {
  [AgentName.RSI]: 1.2,
  [AgentName.MACD]: 1.1,
  [AgentName.Volume]: 1.0,
  [AgentName.Momentum]: 1.15,
  [AgentName.Pattern]: 1.3,
  [AgentName.Sentiment]: 0.9,
  [AgentName.LSTM]: 1.4,
  [AgentName.Options]: 1.25,
  [AgentName.MarketRegime]: 1.0
};
