/**
 * useAgentAnalysis Hook
 * Manages agent workflow analysis with proper error handling and state management
 */

import { useReducer, useCallback, useRef } from 'react';
import { AnalysisState, WorkflowResult, WorkflowStage } from '../types/agent.types';
import { agentWorkflowService } from '../services/agentWorkflowService';
import { debounce } from 'lodash';

// Action types
type AnalysisAction =
  | { type: 'START_ANALYSIS' }
  | { type: 'UPDATE_PROGRESS'; payload: { progress: number; stage: WorkflowStage; message: string } }
  | { type: 'ANALYSIS_SUCCESS'; payload: WorkflowResult }
  | { type: 'ANALYSIS_ERROR'; payload: Error }
  | { type: 'RESET' }
  | { type: 'ADD_MESSAGE'; payload: string };

// Initial state
const initialState: AnalysisState = {
  status: 'idle',
  error: null,
  data: null,
  progress: 0,
  currentStage: null,
  messages: []
};

// Reducer
function analysisReducer(state: AnalysisState, action: AnalysisAction): AnalysisState {
  switch (action.type) {
    case 'START_ANALYSIS':
      return {
        ...initialState,
        status: 'loading',
        progress: 0,
        messages: ['Starting agent analysis...']
      };

    case 'UPDATE_PROGRESS':
      return {
        ...state,
        progress: action.payload.progress,
        currentStage: action.payload.stage,
        messages: [...state.messages, action.payload.message]
      };

    case 'ANALYSIS_SUCCESS':
      return {
        ...state,
        status: 'success',
        data: action.payload,
        progress: 100,
        currentStage: 'complete',
        messages: [...state.messages, 'Analysis complete!']
      };

    case 'ANALYSIS_ERROR':
      return {
        ...state,
        status: 'error',
        error: action.payload,
        progress: 0,
        messages: [...state.messages, `Error: ${action.payload.message}`]
      };

    case 'RESET':
      return initialState;

    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.payload]
      };

    default:
      return state;
  }
}

interface UseAgentAnalysisOptions {
  debounceMs?: number;
  onSuccess?: (result: WorkflowResult) => void;
  onError?: (error: Error) => void;
}

export function useAgentAnalysis(symbol: string, timeframe?: string, options: UseAgentAnalysisOptions = {}) {
  const { debounceMs = 1000, onSuccess, onError } = options;
  const [state, dispatch] = useReducer(analysisReducer, initialState);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Core analysis function
  const analyzeInternal = useCallback(async () => {
    // Cancel any in-progress analysis
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController();

    dispatch({ type: 'START_ANALYSIS' });

    try {
      // Determine if this is day trading or long-term analysis
      const isDayTrading = timeframe && ['1m', '5m', '15m', '30m', '1h'].includes(timeframe);
      const isLongTerm = timeframe && ['1M', '3M', '6M', '1y', '2y', '5y', '10y', 'max'].includes(timeframe);

      // Customize messages based on timeframe
      const progressStages: Array<{ stage: WorkflowStage; progress: number; message: string }> = [
        {
          stage: 'market_regime',
          progress: 15,
          message: isDayTrading ? 'Analyzing intraday volatility...' :
                   isLongTerm ? 'Detecting long-term market cycles...' :
                   'Detecting market regime...'
        },
        {
          stage: 'collecting_signals',
          progress: 30,
          message: isDayTrading ? 'Collecting real-time agent signals...' :
                   isLongTerm ? 'Analyzing fundamental indicators...' :
                   'Collecting agent signals...'
        },
        {
          stage: 'searching_patterns',
          progress: 45,
          message: isDayTrading ? 'Searching intraday patterns...' :
                   isLongTerm ? 'Analyzing historical trends...' :
                   'Searching historical patterns...'
        },
        {
          stage: 'building_consensus',
          progress: 60,
          message: isDayTrading ? 'Building short-term consensus...' :
                   isLongTerm ? 'Evaluating long-term outlook...' :
                   'Building consensus...'
        },
        {
          stage: 'assessing_risk',
          progress: 80,
          message: isDayTrading ? 'Calculating intraday risk limits...' :
                   isLongTerm ? 'Assessing position sizing...' :
                   'Assessing risk parameters...'
        },
        {
          stage: 'making_decision',
          progress: 95,
          message: isDayTrading ? 'Finalizing day trade setup...' :
                   isLongTerm ? 'Confirming investment thesis...' :
                   'Making final decision...'
        }
      ];

      // Start progress simulation
      let progressIndex = 0;
      const progressInterval = setInterval(() => {
        if (progressIndex < progressStages.length && !abortControllerRef.current?.signal.aborted) {
          const stage = progressStages[progressIndex];
          dispatch({
            type: 'UPDATE_PROGRESS',
            payload: stage
          });
          progressIndex++;
        }
      }, 500);

      // Perform the actual analysis with timeframe context
      const result = await agentWorkflowService.analyzeWithWorkflow(symbol, timeframe);

      // Clear progress interval
      clearInterval(progressInterval);

      if (abortControllerRef.current?.signal.aborted) {
        throw new Error('Analysis cancelled');
      }

      if (!result) {
        throw new Error('No analysis result received');
      }

      dispatch({ type: 'ANALYSIS_SUCCESS', payload: result });
      onSuccess?.(result);

    } catch (error) {
      const err = error instanceof Error ? error : new Error('Unknown error occurred');

      if (err.message !== 'Analysis cancelled') {
        dispatch({ type: 'ANALYSIS_ERROR', payload: err });
        onError?.(err);
      }
    } finally {
      abortControllerRef.current = null;
    }
  }, [symbol, timeframe, onSuccess, onError]);

  // Debounced analyze function
  const analyze = useCallback(
    debounce(analyzeInternal, debounceMs),
    [analyzeInternal, debounceMs]
  );

  // Cancel analysis
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      dispatch({ type: 'RESET' });
    }
  }, []);

  // Retry function
  const retry = useCallback(() => {
    dispatch({ type: 'RESET' });
    analyzeInternal();
  }, [analyzeInternal]);

  // Reset state
  const reset = useCallback(() => {
    cancel();
    dispatch({ type: 'RESET' });
  }, [cancel]);

  return {
    // State
    ...state,
    isAnalyzing: state.status === 'loading',
    hasError: state.status === 'error',
    isComplete: state.status === 'success',

    // Actions
    analyze,
    cancel,
    retry,
    reset,

    // Derived data
    tradingLevels: state.data ? agentWorkflowService.extractTradingLevels(state.data.final_decision) : null,
    agentSignals: state.data ? agentWorkflowService.formatAgentSignalsForChart(state.data.agent_signals) : []
  };
}
