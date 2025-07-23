/**
 * LLM Advisor Service
 * Generates human-readable trading advice from signals
 */

import { SignalUpdate } from '../services/websocket/SignalWebSocketManager';
import logger from './logger';


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
  metadata?: {
    pattern?: string;
    indicators?: string[];
    market_context?: string;
  };
}

export interface LLMAnalysisResult {
  advice: TradingAdvice;
  summary: string;
  llm_reasoning: string;
  timestamp: number;
}

class LLMAdvisorService {
  private cache: Map<string, LLMAnalysisResult> = new Map();
  private cacheTimeout = 30000; // 30 seconds cache

  /**
   * Analyze signal with LLM and generate advice
   */
  async analyzeSignalWithLLM(signal: SignalUpdate): Promise<LLMAnalysisResult> {
    const cacheKey = `${signal.symbol}-${signal.signal_id}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached;
    }

    try {
      // Call backend LLM endpoint
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/v1/llm/analyze-signal`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            signal_id: signal.signal_id,
            symbol: signal.symbol,
            signal_type: signal.signal_type,
            confidence: signal.confidence,
            reasoning: signal.reasoning,
            agent_breakdown: signal.agent_breakdown,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`LLM analysis failed: ${response.statusText}`);
      }

      const llmResult = await response.json();

      // Transform to our format
      const advice: TradingAdvice = {
        id: signal.signal_id,
        symbol: signal.symbol,
        action: signal.signal_type,
        entry_price: llmResult.entry_price || signal.metadata?.entry_price || 0,
        stop_loss: llmResult.stop_loss || signal.metadata?.stop_loss || 0,
        take_profits: llmResult.take_profits || signal.metadata?.take_profits || [],
        confidence: signal.confidence,
        reasoning: signal.reasoning || '',
        risk_reward_ratio: llmResult.risk_reward_ratio || 2.0,
        expires_at: Date.now() + 300000, // 5 minutes
        metadata: {
          pattern: llmResult.pattern,
          indicators: llmResult.indicators,
          market_context: llmResult.market_context,
        },
      };

      const result: LLMAnalysisResult = {
        advice,
        summary: llmResult.summary || this.generateFallbackSummary(advice),
        llm_reasoning: llmResult.reasoning || '',
        timestamp: Date.now(),
      };

      // Cache result
      this.cache.set(cacheKey, result);
      return result;

    } catch (error) {
      logger.error('LLM analysis failed:', error);
      // Fallback to local analysis
      return this.generateFallbackAnalysis(signal);
    }
  }

  /**
   * Generate human-readable advice text
   */
  generateAdviceText(advice: TradingAdvice): string {
    const action = advice.action === 'BUY' ? 'Enter BUY' : advice.action === 'SELL' ? 'Enter SELL' : 'HOLD';
    const entryText = advice.entry_price > 0 ? `at $${advice.entry_price.toFixed(2)}` : '';
    const slText = advice.stop_loss > 0 ? `, SL at $${advice.stop_loss.toFixed(2)}` : '';
    const tpText = advice.take_profits.length > 0 ? `, TP at $${advice.take_profits[0].toFixed(2)}` : '';
    const confidenceText = `(${(advice.confidence * 100).toFixed(0)}% confidence)`;

    return `${action} ${advice.symbol} ${entryText}${tpText}${slText} ${confidenceText}`;
  }

  /**
   * Fallback analysis when LLM is unavailable
   */
  private generateFallbackAnalysis(signal: SignalUpdate): LLMAnalysisResult {
    // Extract data from signal metadata
    const currentPrice = signal.metadata?.current_price || 100;
    const atr = signal.metadata?.atr || currentPrice * 0.02;

    const advice: TradingAdvice = {
      id: signal.signal_id,
      symbol: signal.symbol,
      action: signal.signal_type,
      entry_price: currentPrice,
      stop_loss: signal.signal_type === 'BUY'
        ? currentPrice - atr
        : currentPrice + atr,
      take_profits: signal.signal_type === 'BUY'
        ? [currentPrice + atr * 2, currentPrice + atr * 3]
        : [currentPrice - atr * 2, currentPrice - atr * 3],
      confidence: signal.confidence,
      reasoning: signal.reasoning || 'Technical analysis signal',
      risk_reward_ratio: 2.0,
      expires_at: Date.now() + 300000,
    };

    return {
      advice,
      summary: this.generateFallbackSummary(advice),
      llm_reasoning: 'Based on technical indicators and market structure',
      timestamp: Date.now(),
    };
  }

  private generateFallbackSummary(advice: TradingAdvice): string {
    const indicators = advice.metadata?.indicators?.join(', ') || 'RSI, MACD';
    return `Based on ${indicators} analysis, ${advice.action} signal detected with ${(advice.confidence * 100).toFixed(0)}% confidence. Risk/Reward ratio: ${advice.risk_reward_ratio.toFixed(1)}:1`;
  }

  /**
   * Clear expired cache entries
   */
  clearExpiredCache(): void {
    const now = Date.now();
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > this.cacheTimeout) {
        this.cache.delete(key);
      }
    }
  }
}

// Export singleton instance
export const llmAdvisorService = new LLMAdvisorService();

// Clean cache periodically
setInterval(() => {
  llmAdvisorService.clearExpiredCache();
}, 60000);
