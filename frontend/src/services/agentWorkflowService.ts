/**
 * Agent Workflow Service
 *
 * This service integrates with the backend LangGraph workflow and agent orchestration system.
 * It provides methods to:
 * - Analyze symbols using the full agent workflow
 * - Get agent status and performance metrics
 * - Subscribe to real-time agent signals via WebSocket
 * - Format agent data for chart display
 *
 * The service implements caching to reduce API calls and improve performance.
 * All methods include proper error handling and type safety.
 */

import logger from './logger';
import {
  AgentSignal,
  WorkflowDecision,
  WorkflowResult,
  AgentStatus,
  TradingLevels,
  ServiceResponse,
  ChartSignal
} from '../types/agent.types';

/**
 * Main service class for agent workflow operations
 */
class AgentWorkflowService {
  private baseUrl: string;
  // Cache to store recent workflow results and reduce API calls
  private cache: Map<string, { data: WorkflowResult; timestamp: number }> = new Map();
  // Cache duration in milliseconds (30 seconds)
  private cacheDuration = 30000;

  constructor() {
    // Get API URL from environment or use default localhost
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  }

  /**
   * Analyzes a trading symbol using the complete LangGraph workflow
   *
   * This triggers the full agent analysis pipeline:
   * 1. Market regime detection
   * 2. Multi-agent signal collection (8 specialized agents)
   * 3. Historical pattern matching via vector memory
   * 4. Consensus building with weighted voting
   * 5. Risk assessment and position sizing
   * 6. Final decision with Guardrails validation
   *
   * @param symbol - The trading symbol to analyze (e.g., 'AAPL', 'MSFT')
   * @param timeframe - Optional timeframe for context-aware analysis
   * @returns Promise<WorkflowResult | null> - Complete workflow result or null on error
   */
  async analyzeWithWorkflow(symbol: string, timeframe?: string): Promise<WorkflowResult | null> {
    try {
      // Check cache first
      const cacheKey = timeframe ? `${symbol}-${timeframe}` : symbol;
      const cached = this.cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.cacheDuration) {
        return cached.data;
      }

      const response = await fetch(`${this.baseUrl}/api/v1/workflow/analyze-langgraph/${symbol}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ timeframe })
      });

      if (!response.ok) {
        throw new Error(`Workflow analysis failed: ${response.statusText}`);
      }

      const data = await response.json();

      // Cache the result
      this.cache.set(cacheKey, { data, timestamp: Date.now() });

      return data;
    } catch (error) {
      logger.error('Failed to analyze with workflow:', error);
      return null;
    }
  }

  /**
   * Retrieves the current status and performance metrics for all agents
   *
   * This provides insights into:
   * - Which agents are active/offline
   * - Historical accuracy of each agent
   * - Average confidence levels
   * - Performance metrics (signals generated, success rate, etc.)
   *
   * @returns Promise<AgentStatus[]> - Array of agent status objects
   */
  async getAgentsStatus(): Promise<AgentStatus[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/agents/`);

      if (!response.ok) {
        throw new Error(`Failed to fetch agents status: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Failed to get agents status:', error);
      return [];
    }
  }

  /**
   * Get agent performance metrics
   */
  async getAgentPerformance(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/agents/performance`);

      if (!response.ok) {
        throw new Error(`Failed to fetch agent performance: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Failed to get agent performance:', error);
      return {};
    }
  }

  /**
   * Creates a WebSocket subscription to receive real-time agent signals
   *
   * This establishes a persistent connection to receive:
   * - Individual agent signals as they're generated
   * - Workflow progress updates
   * - Agent status changes
   * - Error notifications
   *
   * @param symbol - The trading symbol to monitor
   * @param onSignal - Callback function invoked when new signals arrive
   * @returns Cleanup function to close the WebSocket connection
   *
   * @example
   * const unsubscribe = agentWorkflowService.subscribeToAgentSignals('AAPL', (signal) => {
   *   logger.info('New signal:', signal);
   * });
   *
   * // Later, clean up the connection
   * unsubscribe();
   */
  subscribeToAgentSignals(symbol: string, onSignal: (signal: any) => void): () => void {
    const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/v2/signals/${symbol}`);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'agent_signal') {
          onSignal(data);
        }
      } catch (error) {
        logger.error('Failed to parse agent signal:', error);
      }
    };

    ws.onerror = (error) => {
      logger.error('WebSocket error:', error);
    };

    // Return cleanup function
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }

  /**
   * Extracts chart-drawable trading levels from a workflow decision
   *
   * Transforms the workflow decision into a format suitable for visualization:
   * - Entry price line
   * - Stop loss level (risk management)
   * - Take profit targets (reward levels)
   * - Action type (BUY/SELL) for color coding
   * - Risk assessment for visual indicators
   *
   * @param decision - The workflow decision containing trading parameters
   * @returns TradingLevels object or null if decision is not executable
   */
  extractTradingLevels(decision: WorkflowDecision): TradingLevels | null {
    if (!decision || !decision.execute) {
      return null;
    }

    return {
      entry: decision.entry_price,
      stopLoss: decision.stop_loss,
      takeProfits: [decision.take_profit], // Can be extended for multiple TP levels
      action: decision.action,
      confidence: decision.confidence,
      riskLevel: decision.risk_level
    };
  }

  /**
   * Formats raw agent signals into a structure optimized for chart visualization
   *
   * Filters and transforms agent signals based on:
   * - Confidence threshold (>70% by default)
   * - Signal type (excludes 'hold' signals)
   * - Adds visual properties (color, strength)
   *
   * @param agentSignals - Raw agent signals from workflow
   * @returns Array of formatted signals ready for chart display
   */
  formatAgentSignalsForChart(agentSignals: Record<string, AgentSignal>): ChartSignal[] {
    const signals = [];

    for (const [agentName, signal] of Object.entries(agentSignals)) {
      if (signal.signal !== 'hold' && signal.confidence > 0.7) {
        signals.push({
          agent: agentName,
          type: signal.signal as 'buy' | 'sell',
          confidence: signal.confidence,
          reason: signal.metadata?.reason || `${agentName} signal`,
          strength: signal.confidence
        });
      }
    }

    return signals;
  }

  /**
   * Clear cache for a symbol
   */
  clearCache(symbol?: string) {
    if (symbol) {
      this.cache.delete(symbol);
    } else {
      this.cache.clear();
    }
  }
}

export const agentWorkflowService = new AgentWorkflowService();
