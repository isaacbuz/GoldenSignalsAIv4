import { useEffect, useRef, useState } from 'react';
import { IChartApi } from 'lightweight-charts';
import { ChartSignalAgent, SignalOverlay, ChartSignal } from '../agents/ChartSignalAgent';
import logger from '../services/logger';


interface UseChartSignalAgentProps {
  chart: IChartApi | null;
  symbol: string;
  timeframe: string;
  wsUrl?: string;
  enabled?: boolean;
}

export function useChartSignalAgent({
  chart,
  symbol,
  timeframe,
  wsUrl = 'ws://localhost:8000/ws',
  enabled = true,
}: UseChartSignalAgentProps) {
  const agentRef = useRef<ChartSignalAgent | null>(null);
  const [signals, setSignals] = useState<SignalOverlay[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!enabled) return;

    // Create agent instance
    const agent = new ChartSignalAgent({
      chart: null, // Start with null, will be set when chart is ready
      updateInterval: 50,
      showConfidenceThreshold: 0.75,
      enableRealTimeUpdates: true, // Enable WebSocket for real-time updates
    });

    agentRef.current = agent;

    // Subscribe to signal updates
    const subscription = agent.getSignals().subscribe(newSignals => {
      setSignals(newSignals);
    });

    // Connect to backend if URL provided
    if (wsUrl) {
      try {
        agent.connectToBackend(`${wsUrl}?symbol=${symbol}&timeframe=${timeframe}`);
        setIsConnected(true);
      } catch (error) {
        logger.error('Failed to connect ChartSignalAgent:', error);
        setIsConnected(false);
      }
    }

    return () => {
      subscription.unsubscribe();
      agent.destroy();
      setIsConnected(false);
    };
  }, [symbol, timeframe, wsUrl, enabled]); // Removed chart from dependencies

  // Update chart reference when it changes
  useEffect(() => {
    if (agentRef.current && chart) {
      logger.info('Setting chart in agent', chart);
      agentRef.current.setChart(chart);

      // Add test signals to verify positioning
      if (signals.length === 0) {
        // These will be replaced by real signals from loadData
        logger.info('Agent ready for signals');
      }
    }
  }, [chart]);

  // Public methods to control the agent
  const addSignal = (signal: ChartSignal) => {
    agentRef.current?.addSignal(signal);
  };

  const removeSignal = (signalId: string) => {
    agentRef.current?.removeSignal(signalId);
  };

  const clearSignals = () => {
    agentRef.current?.clearSignals();
  };

  const updateFromConsensus = (consensus: any) => {
    agentRef.current?.updateSignalsFromConsensus(consensus);
  };

  return {
    signals,
    isConnected,
    addSignal,
    removeSignal,
    clearSignals,
    updateFromConsensus,
    agent: agentRef.current,
  };
}
