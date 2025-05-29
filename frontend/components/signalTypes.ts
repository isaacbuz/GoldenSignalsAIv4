// TypeScript interfaces for signal logs and analysis results

export interface AgentData {
  explanation: string;
  [key: string]: any;
}

export interface BlendedData {
  strategy: string;
  confidence: number;
  explanation: string;
  [key: string]: any;
}

export interface SignalLog {
  timestamp: string;
  ticker: string;
  blended: BlendedData;
  agents: Record<string, AgentData>;
}

export interface AnalysisResult {
  strategy: string;
  entry: number;
  exit: number;
  confidence: number;
  explanation: string;
  supporting_signals: Record<string, AgentData>;
}

export interface PricePoint {
  time: number | string;
  value: number;
}

export interface ForecastTrendPoint {
  time: number | string;
  value: number;
}
