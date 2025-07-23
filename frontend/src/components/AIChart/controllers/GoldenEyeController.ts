/**
 * Golden Eye Chart Controller
 *
 * Specialized controller for Golden Eye AI Prophet Chat integration.
 * Extends the base ChartController with Golden Eye specific functionality.
 */

import { ChartController, IChartController, PredictionParams, AgentSignal, ChartPattern } from './ChartController';
import { ChartAction } from '../../GoldenEyeChat/GoldenEyeChat';
import logger from '../../../services/logger';


interface GoldenEyePrediction extends PredictionParams {
  llmSource: 'openai' | 'anthropic' | 'grok';
  agentsConsulted: string[];
  consensus: {
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    votes: Record<string, string>;
  };
}

interface AgentCluster {
  centroid: { time: number; price: number };
  signals: AgentSignal[];
  consensus: string;
  strength: number;
}

export interface IGoldenEyeController extends IChartController {
  // Golden Eye specific methods
  executeChartAction(action: ChartAction): Promise<void>;
  showAgentConsensus(consensus: any): void;
  highlightAgentActivity(agentName: string, duration?: number): void;
  showPredictionConfidence(confidence: number): void;

  // Multi-agent visualization
  clusterAgentSignals(threshold?: number): AgentCluster[];
  showAgentDisagreement(agents: string[]): void;
  animateAgentThinking(agentName: string): void;

  // Advanced predictions
  drawMultiModelPrediction(predictions: GoldenEyePrediction[]): void;
  showPredictionDivergence(predictions: GoldenEyePrediction[]): void;

  // Interactive features
  enableClickToAnalyze(): void;
  disableClickToAnalyze(): void;
  enableDragToSelectTimeRange(): void;

  // Real-time updates
  subscribeToAgentUpdates(agentName: string): void;
  unsubscribeFromAgentUpdates(agentName: string): void;
}

export class GoldenEyeController extends ChartController implements IGoldenEyeController {
  private activeSubscriptions: Set<string> = new Set();
  private clickToAnalyzeEnabled = false;
  private dragSelectEnabled = false;
  private agentActivityTimers: Map<string, NodeJS.Timeout> = new Map();

  constructor(chartRef: any) {
    super(chartRef);
    this.setupGoldenEyeFeatures();
  }

  private setupGoldenEyeFeatures(): void {
    // Add Golden Eye specific event listeners
    this.on('chartClick', this.handleChartClick.bind(this));
    this.on('chartDragSelect', this.handleDragSelect.bind(this));
  }

  /**
   * Execute a chart action from Golden Eye Chat
   */
  async executeChartAction(action: ChartAction): Promise<void> {
    switch (action.type) {
      case 'draw_prediction':
        await this.handleDrawPrediction(action.data);
        break;

      case 'add_agent_signals':
        this.handleAddAgentSignals(action.data);
        break;

      case 'mark_entry_point':
        this.handleMarkEntryPoint(action.data);
        break;

      case 'mark_exit_point':
        this.handleMarkExitPoint(action.data);
        break;

      case 'draw_levels':
        this.handleDrawLevels(action.data);
        break;

      case 'highlight_pattern':
        this.handleHighlightPattern(action.data);
        break;

      default:
        logger.warn(`Unknown chart action type: ${action.type}`);
    }
  }

  /**
   * Show agent consensus visualization
   */
  showAgentConsensus(consensus: any): void {
    this.emit('showConsensus', {
      consensus,
      position: 'top-right',
      animation: 'fadeIn'
    });

    // Add consensus marker on chart
    if (consensus.recommendedPrice) {
      this.addAnnotation({
        id: `consensus_${Date.now()}`,
        type: 'text',
        data: {
          x: Date.now(),
          y: consensus.recommendedPrice,
          text: `Consensus: ${consensus.action} (${(consensus.confidence * 100).toFixed(0)}%)`,
          style: {
            background: consensus.action === 'BUY' ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 68, 68, 0.2)',
            borderColor: consensus.action === 'BUY' ? '#00FF88' : '#FF4444',
            borderWidth: 2,
            padding: 8,
            borderRadius: 4
          }
        }
      });
    }
  }

  /**
   * Highlight specific agent's activity on the chart
   */
  highlightAgentActivity(agentName: string, duration = 3000): void {
    // Clear any existing highlight for this agent
    const existingTimer = this.agentActivityTimers.get(agentName);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Highlight all signals from this agent
    this.highlightSignals(agentName);

    // Add agent activity indicator
    this.emit('showAgentActivity', {
      agentName,
      animation: 'pulse'
    });

    // Auto-clear after duration
    const timer = setTimeout(() => {
      this.emit('hideAgentActivity', { agentName });
      this.agentActivityTimers.delete(agentName);
    }, duration);

    this.agentActivityTimers.set(agentName, timer);
  }

  /**
   * Show prediction confidence visualization
   */
  showPredictionConfidence(confidence: number): void {
    const level = confidence > 0.8 ? 'high' : confidence > 0.6 ? 'medium' : 'low';
    const color = confidence > 0.8 ? '#00FF88' : confidence > 0.6 ? '#FFD700' : '#FF6B6B';

    this.emit('showConfidence', {
      confidence,
      level,
      color,
      position: 'bottom-right'
    });
  }

  /**
   * Cluster nearby agent signals for cleaner visualization
   */
  clusterAgentSignals(threshold = 0.02): AgentCluster[] {
    const signals = Array.from(this['signals'].values());
    const clusters: AgentCluster[] = [];
    const processed = new Set<string>();

    signals.forEach(signal => {
      if (processed.has(signal.id)) return;

      // Find nearby signals
      const cluster: AgentSignal[] = [signal];
      processed.add(signal.id);

      signals.forEach(other => {
        if (processed.has(other.id)) return;

        const priceDiff = Math.abs(signal.price - other.price) / signal.price;
        const timeDiff = Math.abs(signal.time - other.time) / (1000 * 60 * 60); // hours

        if (priceDiff < threshold && timeDiff < 1) {
          cluster.push(other);
          processed.add(other.id);
        }
      });

      // Calculate cluster properties
      const avgPrice = cluster.reduce((sum, s) => sum + s.price, 0) / cluster.length;
      const avgTime = cluster.reduce((sum, s) => sum + s.time, 0) / cluster.length;

      const buyCount = cluster.filter(s => s.type === 'buy').length;
      const sellCount = cluster.filter(s => s.type === 'sell').length;

      clusters.push({
        centroid: { time: avgTime, price: avgPrice },
        signals: cluster,
        consensus: buyCount > sellCount ? 'BUY' : sellCount > buyCount ? 'SELL' : 'HOLD',
        strength: Math.max(buyCount, sellCount) / cluster.length
      });
    });

    return clusters;
  }

  /**
   * Show visual disagreement between agents
   */
  showAgentDisagreement(agents: string[]): void {
    const signals = Array.from(this['signals'].values())
      .filter(s => agents.includes(s.agentName));

    // Group by agent
    const agentGroups = new Map<string, AgentSignal[]>();
    signals.forEach(signal => {
      const group = agentGroups.get(signal.agentName) || [];
      group.push(signal);
      agentGroups.set(signal.agentName, group);
    });

    // Visualize disagreement
    this.emit('showDisagreement', {
      agents: Array.from(agentGroups.entries()).map(([agent, signals]) => ({
        agent,
        signals,
        position: this.calculateAgentPosition(agent, agents)
      })),
      animation: 'diverge'
    });
  }

  /**
   * Animate agent thinking process
   */
  animateAgentThinking(agentName: string): void {
    this.emit('agentThinking', {
      agentName,
      animation: 'pulse',
      duration: 2000
    });
  }

  /**
   * Draw predictions from multiple models
   */
  drawMultiModelPrediction(predictions: GoldenEyePrediction[]): void {
    predictions.forEach((prediction, index) => {
      const opacity = 0.3 + (0.5 / predictions.length);
      const color = this.getLLMColor(prediction.llmSource);

      this.drawPrediction({
        ...prediction,
        metadata: {
          ...prediction.metadata,
          style: {
            lineColor: color,
            lineWidth: 2,
            opacity,
            dashArray: index > 0 ? [5, 5] : undefined
          }
        }
      });
    });

    // Show divergence if predictions differ significantly
    this.showPredictionDivergence(predictions);
  }

  /**
   * Show divergence between predictions
   */
  showPredictionDivergence(predictions: GoldenEyePrediction[]): void {
    if (predictions.length < 2) return;

    // Calculate divergence at each point
    const divergencePoints: Array<{ time: number; divergence: number }> = [];

    const pointCount = predictions[0].points.length;
    for (let i = 0; i < pointCount; i++) {
      const prices = predictions.map(p => p.points[i]?.price || 0);
      const avg = prices.reduce((sum, p) => sum + p, 0) / prices.length;
      const variance = prices.reduce((sum, p) => sum + Math.pow(p - avg, 2), 0) / prices.length;
      const stdDev = Math.sqrt(variance);
      const divergence = stdDev / avg; // Coefficient of variation

      divergencePoints.push({
        time: predictions[0].points[i].time,
        divergence
      });
    }

    // Highlight high divergence areas
    this.emit('showDivergence', {
      points: divergencePoints,
      threshold: 0.02, // 2% divergence
      style: {
        fillColor: 'rgba(255, 107, 107, 0.2)',
        strokeColor: '#FF6B6B'
      }
    });
  }

  /**
   * Enable click-to-analyze feature
   */
  enableClickToAnalyze(): void {
    this.clickToAnalyzeEnabled = true;
    this.emit('featureEnabled', { feature: 'clickToAnalyze' });
  }

  /**
   * Disable click-to-analyze feature
   */
  disableClickToAnalyze(): void {
    this.clickToAnalyzeEnabled = false;
    this.emit('featureDisabled', { feature: 'clickToAnalyze' });
  }

  /**
   * Enable drag to select time range
   */
  enableDragToSelectTimeRange(): void {
    this.dragSelectEnabled = true;
    this.emit('featureEnabled', { feature: 'dragSelect' });
  }

  /**
   * Subscribe to real-time agent updates
   */
  subscribeToAgentUpdates(agentName: string): void {
    if (this.activeSubscriptions.has(agentName)) return;

    this.activeSubscriptions.add(agentName);
    this.emit('subscribeAgent', { agentName });
  }

  /**
   * Unsubscribe from agent updates
   */
  unsubscribeFromAgentUpdates(agentName: string): void {
    this.activeSubscriptions.delete(agentName);
    this.emit('unsubscribeAgent', { agentName });
  }

  // Private helper methods

  private async handleDrawPrediction(data: any): Promise<void> {
    const predictionParams: PredictionParams = {
      symbol: data.symbol,
      points: data.prediction.map((price: number, index: number) => ({
        time: Date.now() + (index * 60 * 60 * 1000), // Hourly points
        price
      })),
      confidence: data.confidence_bands,
      horizon: data.horizon,
      metadata: {
        model: 'ensemble',
        confidence: data.confidence || 0.75,
        factors: data.supporting_factors || []
      }
    };

    await this.drawPrediction(predictionParams);
    this.showPredictionConfidence(predictionParams.metadata?.confidence || 0.75);
  }

  private handleAddAgentSignals(data: any): void {
    data.signals.forEach((signal: any) => {
      this.addAgentSignal({
        id: `signal_${signal.agent}_${Date.now()}`,
        agentName: signal.agent,
        time: signal.timestamp || Date.now(),
        price: signal.price || this.getCurrentPrice(),
        type: signal.signal.toLowerCase() as 'buy' | 'sell' | 'hold',
        confidence: signal.confidence,
        reasoning: signal.analysis?.reasoning
      });
    });

    // Show consensus if available
    if (data.consensus) {
      this.showAgentConsensus(data.consensus);
    }
  }

  private handleMarkEntryPoint(data: any): void {
    this.drawEntryExitZones([{
      startPrice: data.price - (data.price * 0.002), // 0.2% zone
      endPrice: data.price + (data.price * 0.002),
      type: 'entry',
      label: 'Entry Zone',
      color: '#00FF88',
      opacity: 0.3
    }]);

    this.addAnnotation({
      id: `entry_${Date.now()}`,
      type: 'arrow',
      data: {
        x: Date.now(),
        y: data.price,
        direction: 'up',
        text: 'Entry Signal',
        style: {
          color: '#00FF88',
          size: 20,
          strokeWidth: 3
        }
      }
    });
  }

  private handleMarkExitPoint(data: any): void {
    this.drawEntryExitZones([{
      startPrice: data.price - (data.price * 0.002),
      endPrice: data.price + (data.price * 0.002),
      type: 'exit',
      label: 'Exit Zone',
      color: '#FF4444',
      opacity: 0.3
    }]);

    this.addAnnotation({
      id: `exit_${Date.now()}`,
      type: 'arrow',
      data: {
        x: Date.now(),
        y: data.price,
        direction: 'down',
        text: 'Exit Signal',
        style: {
          color: '#FF4444',
          size: 20,
          strokeWidth: 3
        }
      }
    });
  }

  private handleDrawLevels(data: any): void {
    const levels = data.levels.map((level: any) => ({
      price: level.price,
      type: level.type as 'support' | 'resistance',
      strength: level.strength || 0.5,
      touches: level.touches || 0,
      label: level.label || `${level.type} - $${level.price.toFixed(2)}`
    }));

    this.drawSupportResistance(levels);
  }

  private handleHighlightPattern(data: any): void {
    data.patterns.forEach((pattern: any) => {
      this.highlightPattern({
        id: `pattern_${Date.now()}`,
        name: pattern.name,
        type: pattern.type,
        startTime: pattern.startTime,
        endTime: pattern.endTime,
        keyPoints: pattern.keyPoints || [],
        confidence: pattern.confidence || 0.7
      });
    });
  }

  private handleChartClick(event: any): void {
    if (!this.clickToAnalyzeEnabled) return;

    const { time, price } = event;
    this.emit('analyzePoint', { time, price });
  }

  private handleDragSelect(event: any): void {
    if (!this.dragSelectEnabled) return;

    const { startTime, endTime, startPrice, endPrice } = event;
    this.emit('analyzeRange', { startTime, endTime, startPrice, endPrice });
  }

  private getLLMColor(llm: string): string {
    const colors = {
      openai: '#00A67E',
      anthropic: '#6B46C1',
      grok: '#1DA1F2'
    };
    return colors[llm as keyof typeof colors] || '#FFD700';
  }

  private calculateAgentPosition(agent: string, allAgents: string[]): { x: number; y: number } {
    const index = allAgents.indexOf(agent);
    const angle = (index / allAgents.length) * Math.PI * 2;
    const radius = 100;

    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius
    };
  }
}

// Export factory function
export function createGoldenEyeController(chartRef: any): IGoldenEyeController {
  return new GoldenEyeController(chartRef);
}
