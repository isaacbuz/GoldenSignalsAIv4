/**
 * ChartContext
 *
 * Provides centralized state management for the AI Trading Chart.
 * This context manages all chart-related state including:
 * - Chart data and settings
 * - Agent analysis results
 * - WebSocket connections
 * - User preferences
 *
 * Using React Context prevents prop drilling and makes state
 * accessible to all child components efficiently.
 */

import React, { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';
import {
  WorkflowResult,
  TradingLevels,
  AnalysisState,
  AgentSignal,
  WorkflowStage,
} from '../../../types/agent.types';

/**
 * Chart display settings
 */
interface ChartSettings {
  chartType: 'candlestick' | 'mountain' | 'bars' | 'line';
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  selectedIndicators: string[];
  showVolume: boolean;
  showGrid: boolean;
  theme: 'light' | 'dark';
}

/**
 * Current chart data
 */
interface ChartData {
  symbol: string;
  data: Array<{
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
  loading: boolean;
  error: Error | null;
}

/**
 * Complete chart state
 */
interface ChartState {
  // Chart data
  chartData: ChartData;

  // Display settings
  settings: ChartSettings;

  // Agent analysis
  analysis: AnalysisState;
  agentSignals: Record<string, AgentSignal>;
  workflowResult: WorkflowResult | null;
  tradingLevels: TradingLevels | null;

  // UI state
  showAgentPanel: boolean;
  isConnected: boolean;
  activeTab: number;
  compareSymbols: string[];
}

/**
 * Action types for state updates
 */
type ChartAction =
  | { type: 'SET_SYMBOL'; payload: string }
  | { type: 'SET_CHART_DATA'; payload: ChartData['data'] }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: Error | null }
  | { type: 'UPDATE_SETTINGS'; payload: Partial<ChartSettings> }
  | { type: 'SET_ANALYSIS_STATE'; payload: AnalysisState }
  | { type: 'SET_AGENT_SIGNALS'; payload: Record<string, AgentSignal> }
  | { type: 'SET_WORKFLOW_RESULT'; payload: WorkflowResult | null }
  | { type: 'SET_TRADING_LEVELS'; payload: TradingLevels | null }
  | { type: 'SET_AGENT_PANEL_VISIBLE'; payload: boolean }
  | { type: 'SET_CONNECTION_STATUS'; payload: boolean }
  | { type: 'SET_ACTIVE_TAB'; payload: number }
  | { type: 'ADD_COMPARE_SYMBOL'; payload: string }
  | { type: 'REMOVE_COMPARE_SYMBOL'; payload: string }
  | { type: 'RESET_ANALYSIS' };

/**
 * Initial state values
 */
const initialState: ChartState = {
  chartData: {
    symbol: 'AAPL',
    data: [],
    loading: false,
    error: null,
  },
  settings: {
    chartType: 'candlestick',
    timeframe: '5m',
    selectedIndicators: ['ai-signals', 'ai-predictions'],
    showVolume: true,
    showGrid: true,
    theme: 'dark',
  },
  analysis: {
    status: 'idle',
    error: null,
    data: null,
    progress: 0,
    currentStage: null,
    messages: [],
  },
  agentSignals: {},
  workflowResult: null,
  tradingLevels: null,
  showAgentPanel: false,
  isConnected: false,
  activeTab: 0,
  compareSymbols: [],
};

/**
 * Reducer to handle state updates
 */
function chartReducer(state: ChartState, action: ChartAction): ChartState {
  switch (action.type) {
    case 'SET_SYMBOL':
      return {
        ...state,
        chartData: {
          ...state.chartData,
          symbol: action.payload,
        },
      };

    case 'SET_CHART_DATA':
      return {
        ...state,
        chartData: {
          ...state.chartData,
          data: action.payload,
          loading: false,
          error: null,
        },
      };

    case 'SET_LOADING':
      return {
        ...state,
        chartData: {
          ...state.chartData,
          loading: action.payload,
        },
      };

    case 'SET_ERROR':
      return {
        ...state,
        chartData: {
          ...state.chartData,
          error: action.payload,
          loading: false,
        },
      };

    case 'UPDATE_SETTINGS':
      return {
        ...state,
        settings: {
          ...state.settings,
          ...action.payload,
        },
      };

    case 'SET_ANALYSIS_STATE':
      return {
        ...state,
        analysis: action.payload,
      };

    case 'SET_AGENT_SIGNALS':
      return {
        ...state,
        agentSignals: action.payload,
      };

    case 'SET_WORKFLOW_RESULT':
      return {
        ...state,
        workflowResult: action.payload,
      };

    case 'SET_TRADING_LEVELS':
      return {
        ...state,
        tradingLevels: action.payload,
      };

    case 'SET_AGENT_PANEL_VISIBLE':
      return {
        ...state,
        showAgentPanel: action.payload,
      };

    case 'SET_CONNECTION_STATUS':
      return {
        ...state,
        isConnected: action.payload,
      };

    case 'SET_ACTIVE_TAB':
      return {
        ...state,
        activeTab: action.payload,
      };

    case 'ADD_COMPARE_SYMBOL':
      if (state.compareSymbols.includes(action.payload)) {
        return state;
      }
      return {
        ...state,
        compareSymbols: [...state.compareSymbols, action.payload].slice(0, 3), // Max 3 symbols
      };

    case 'REMOVE_COMPARE_SYMBOL':
      return {
        ...state,
        compareSymbols: state.compareSymbols.filter(s => s !== action.payload),
      };

    case 'RESET_ANALYSIS':
      return {
        ...state,
        analysis: initialState.analysis,
        agentSignals: {},
        workflowResult: null,
        tradingLevels: null,
        showAgentPanel: false,
      };

    default:
      return state;
  }
}

/**
 * Context value interface
 */
interface ChartContextValue {
  state: ChartState;
  dispatch: React.Dispatch<ChartAction>;

  // Chart data actions
  setSymbol: (symbol: string) => void;
  setChartData: (data: ChartData['data']) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: Error | null) => void;

  // Settings actions
  updateSettings: (settings: Partial<ChartSettings>) => void;

  // Analysis actions
  setAnalysisState: (state: AnalysisState) => void;
  setAgentSignals: (signals: Record<string, AgentSignal>) => void;
  setWorkflowResult: (result: WorkflowResult | null) => void;
  setTradingLevels: (levels: TradingLevels | null) => void;
  resetAnalysis: () => void;

  // UI actions
  setAgentPanelVisible: (visible: boolean) => void;
  setConnectionStatus: (connected: boolean) => void;
  setActiveTab: (tab: number) => void;
  addCompareSymbol: (symbol: string) => void;
  removeCompareSymbol: (symbol: string) => void;
}

/**
 * Create the context
 */
const ChartContext = createContext<ChartContextValue | undefined>(undefined);

/**
 * Provider component props
 */
interface ChartProviderProps {
  children: ReactNode;
  initialSymbol?: string;
}

/**
 * Chart Context Provider
 *
 * Wraps the chart component tree and provides state management
 */
export const ChartProvider: React.FC<ChartProviderProps> = ({
  children,
  initialSymbol = 'AAPL'
}) => {
  // Initialize state with custom symbol if provided
  const [state, dispatch] = useReducer(chartReducer, {
    ...initialState,
    chartData: {
      ...initialState.chartData,
      symbol: initialSymbol,
    },
  });

  // Chart data actions
  const setSymbol = useCallback((symbol: string) => {
    dispatch({ type: 'SET_SYMBOL', payload: symbol });
  }, []);

  const setChartData = useCallback((data: ChartData['data']) => {
    dispatch({ type: 'SET_CHART_DATA', payload: data });
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  }, []);

  const setError = useCallback((error: Error | null) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  }, []);

  // Settings actions
  const updateSettings = useCallback((settings: Partial<ChartSettings>) => {
    dispatch({ type: 'UPDATE_SETTINGS', payload: settings });
  }, []);

  // Analysis actions
  const setAnalysisState = useCallback((analysisState: AnalysisState) => {
    dispatch({ type: 'SET_ANALYSIS_STATE', payload: analysisState });
  }, []);

  const setAgentSignals = useCallback((signals: Record<string, AgentSignal>) => {
    dispatch({ type: 'SET_AGENT_SIGNALS', payload: signals });
  }, []);

  const setWorkflowResult = useCallback((result: WorkflowResult | null) => {
    dispatch({ type: 'SET_WORKFLOW_RESULT', payload: result });
  }, []);

  const setTradingLevels = useCallback((levels: TradingLevels | null) => {
    dispatch({ type: 'SET_TRADING_LEVELS', payload: levels });
  }, []);

  const resetAnalysis = useCallback(() => {
    dispatch({ type: 'RESET_ANALYSIS' });
  }, []);

  // UI actions
  const setAgentPanelVisible = useCallback((visible: boolean) => {
    dispatch({ type: 'SET_AGENT_PANEL_VISIBLE', payload: visible });
  }, []);

  const setConnectionStatus = useCallback((connected: boolean) => {
    dispatch({ type: 'SET_CONNECTION_STATUS', payload: connected });
  }, []);

  const setActiveTab = useCallback((tab: number) => {
    dispatch({ type: 'SET_ACTIVE_TAB', payload: tab });
  }, []);

  const addCompareSymbol = useCallback((symbol: string) => {
    dispatch({ type: 'ADD_COMPARE_SYMBOL', payload: symbol });
  }, []);

  const removeCompareSymbol = useCallback((symbol: string) => {
    dispatch({ type: 'REMOVE_COMPARE_SYMBOL', payload: symbol });
  }, []);

  // Context value
  const value: ChartContextValue = {
    state,
    dispatch,
    setSymbol,
    setChartData,
    setLoading,
    setError,
    updateSettings,
    setAnalysisState,
    setAgentSignals,
    setWorkflowResult,
    setTradingLevels,
    resetAnalysis,
    setAgentPanelVisible,
    setConnectionStatus,
    setActiveTab,
    addCompareSymbol,
    removeCompareSymbol,
  };

  return (
    <ChartContext.Provider value={value}>
      {children}
    </ChartContext.Provider>
  );
};

/**
 * Custom hook to use the Chart Context
 *
 * @throws Error if used outside of ChartProvider
 */
export const useChartContext = (): ChartContextValue => {
  const context = useContext(ChartContext);
  if (!context) {
    throw new Error('useChartContext must be used within a ChartProvider');
  }
  return context;
};
