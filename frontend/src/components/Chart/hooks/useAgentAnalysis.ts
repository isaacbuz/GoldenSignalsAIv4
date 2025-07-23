/**
 * Agent analysis hook
 * Connects to the multi-agent system for AI-powered analysis
 */

import { useState, useEffect, useCallback } from 'react';

export interface AgentSignal {
  agent: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  timestamp: string;
}

export interface ConsensusDecision {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  entry_price?: number;
  stop_loss?: number;
  take_profit?: number;
  risk_score: number;
  supporting_agents: number;
  opposing_agents: number;
}

interface UseAgentAnalysisReturn {
  agentSignals: AgentSignal[] | null;
  consensus: ConsensusDecision | null;
  isAnalyzing: boolean;
  error: string | null;
  triggerAnalysis: () => void;
}

export function useAgentAnalysis(symbol: string): UseAgentAnalysisReturn {
  const [agentSignals, setAgentSignals] = useState<AgentSignal[] | null>(null);
  const [consensus, setConsensus] = useState<ConsensusDecision | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Trigger analysis
  const triggerAnalysis = useCallback(async () => {
    try {
      setIsAnalyzing(true);
      setError(null);

      const response = await fetch(
        `http://localhost:8000/api/v1/workflow/analyze-langgraph/${symbol}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error('Failed to analyze');
      }

      const data = await response.json();

      // Extract agent signals
      const signals: AgentSignal[] = data.agent_signals?.map((signal: any) => ({
        agent: signal.agent_name,
        signal: signal.signal,
        confidence: signal.confidence,
        reasoning: signal.reasoning || '',
        timestamp: signal.timestamp || new Date().toISOString(),
      })) || [];

      setAgentSignals(signals);

      // Extract consensus
      if (data.consensus) {
        setConsensus({
          signal: data.consensus.final_signal,
          confidence: data.consensus.confidence,
          entry_price: data.consensus.entry_price,
          stop_loss: data.consensus.stop_loss,
          take_profit: data.consensus.take_profit,
          risk_score: data.consensus.risk_score || 0.5,
          supporting_agents: data.consensus.supporting_agents || 0,
          opposing_agents: data.consensus.opposing_agents || 0,
        });
      }

    } catch (err) {
      console.error('Agent analysis error:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [symbol]);

  // Auto-analyze on symbol change
  useEffect(() => {
    // Reset state
    setAgentSignals(null);
    setConsensus(null);
    setError(null);

    // Trigger initial analysis after a delay
    const timeout = setTimeout(() => {
      triggerAnalysis();
    }, 2000);

    return () => clearTimeout(timeout);
  }, [symbol, triggerAnalysis]);

  // Periodic re-analysis
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isAnalyzing) {
        triggerAnalysis();
      }
    }, 5 * 60 * 1000); // Every 5 minutes

    return () => clearInterval(interval);
  }, [isAnalyzing, triggerAnalysis]);

  return {
    agentSignals,
    consensus,
    isAnalyzing,
    error,
    triggerAnalysis,
  };
}
