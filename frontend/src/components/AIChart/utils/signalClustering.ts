/**
 * Signal Clustering Utility
 *
 * Groups nearby agent signals to prevent overlapping visualizations
 * and create cleaner chart displays.
 */

import { AgentSignal } from '../components/ChartCanvas/types';

export interface SignalCluster {
  id: string;
  signals: AgentSignal[];
  time: number; // Average time
  price: number; // Average price
  type: 'buy' | 'sell' | 'mixed';
  buyCount: number;
  sellCount: number;
  consensus: 'BUY' | 'SELL' | 'HOLD';
  strength: number; // 0-1 based on confidence and agreement
  agents: string[]; // List of agent names
}

/**
 * Cluster nearby signals based on price and time proximity
 */
export function clusterSignals(
  signals: AgentSignal[],
  priceThreshold: number = 0.001, // 0.1% price difference
  timeThreshold: number = 60000    // 1 minute
): SignalCluster[] {
  if (signals.length === 0) return [];

  const clusters: SignalCluster[] = [];
  const used = new Set<string>();

  // Sort signals by time for better clustering
  const sortedSignals = [...signals].sort((a, b) => a.time - b.time);

  sortedSignals.forEach(signal => {
    if (used.has(signal.id)) return;

    // Start new cluster with this signal
    const cluster: AgentSignal[] = [signal];
    used.add(signal.id);

    // Find nearby signals
    sortedSignals.forEach(other => {
      if (used.has(other.id)) return;

      const priceDiff = Math.abs(signal.price - other.price) / signal.price;
      const timeDiff = Math.abs(signal.time - other.time);

      if (priceDiff <= priceThreshold && timeDiff <= timeThreshold) {
        cluster.push(other);
        used.add(other.id);
      }
    });

    // Create cluster from grouped signals
    clusters.push(createClusterFromSignals(cluster));
  });

  return clusters;
}

/**
 * Create a cluster summary from grouped signals
 */
function createClusterFromSignals(signals: AgentSignal[]): SignalCluster {
  const buySignals = signals.filter(s => s.type === 'buy');
  const sellSignals = signals.filter(s => s.type === 'sell');

  // Calculate average position
  const avgTime = signals.reduce((sum, s) => sum + s.time, 0) / signals.length;
  const avgPrice = signals.reduce((sum, s) => sum + s.price, 0) / signals.length;

  // Calculate weighted average confidence
  const totalConfidence = signals.reduce((sum, s) => sum + s.confidence, 0);
  const avgConfidence = totalConfidence / signals.length;

  // Determine consensus
  let consensus: 'BUY' | 'SELL' | 'HOLD';
  let strength: number;

  if (buySignals.length > sellSignals.length * 2) {
    consensus = 'BUY';
    strength = (buySignals.length / signals.length) * avgConfidence;
  } else if (sellSignals.length > buySignals.length * 2) {
    consensus = 'SELL';
    strength = (sellSignals.length / signals.length) * avgConfidence;
  } else {
    consensus = 'HOLD';
    strength = avgConfidence * 0.5; // Lower strength for mixed signals
  }

  // Extract unique agent names
  const agents = [...new Set(signals.map(s => s.agentName))];

  return {
    id: `cluster-${avgTime}-${avgPrice}`,
    signals,
    time: avgTime,
    price: avgPrice,
    type: buySignals.length > sellSignals.length ? 'buy' :
          sellSignals.length > buySignals.length ? 'sell' : 'mixed',
    buyCount: buySignals.length,
    sellCount: sellSignals.length,
    consensus,
    strength,
    agents,
  };
}

/**
 * Filter clusters by minimum signal count
 */
export function filterClustersBySize(
  clusters: SignalCluster[],
  minSignals: number = 2
): SignalCluster[] {
  return clusters.filter(cluster => cluster.signals.length >= minSignals);
}

/**
 * Sort clusters by strength (highest first)
 */
export function sortClustersByStrength(clusters: SignalCluster[]): SignalCluster[] {
  return [...clusters].sort((a, b) => b.strength - a.strength);
}

/**
 * Get cluster color based on consensus and strength
 */
export function getClusterColor(
  cluster: SignalCluster,
  theme: 'light' | 'dark'
): string {
  const baseColors = {
    BUY: theme === 'dark' ? '#00FF88' : '#4CAF50',
    SELL: theme === 'dark' ? '#FF4444' : '#F44336',
    HOLD: theme === 'dark' ? '#FFD700' : '#FFC107',
  };

  const color = baseColors[cluster.consensus];
  const opacity = Math.round(cluster.strength * 255).toString(16).padStart(2, '0');

  return color + opacity;
}

/**
 * Calculate cluster importance score for prioritization
 */
export function getClusterImportance(cluster: SignalCluster): number {
  // Factors:
  // - Number of agreeing agents
  // - Average confidence
  // - Consensus strength

  const agentScore = cluster.signals.length / 9; // Normalize by total agents
  const confidenceScore = cluster.strength;
  const consensusScore = cluster.consensus === 'HOLD' ? 0.5 : 1;

  return (agentScore + confidenceScore + consensusScore) / 3;
}

/**
 * Merge overlapping clusters
 */
export function mergeOverlappingClusters(
  clusters: SignalCluster[],
  overlapThreshold: number = 0.8
): SignalCluster[] {
  const merged: SignalCluster[] = [];
  const used = new Set<string>();

  clusters.forEach(cluster => {
    if (used.has(cluster.id)) return;

    // Check for overlaps with other clusters
    const overlapping = clusters.filter(other => {
      if (other.id === cluster.id || used.has(other.id)) return false;

      // Check if clusters share many agents
      const sharedAgents = cluster.agents.filter(a => other.agents.includes(a));
      const overlapRatio = sharedAgents.length / Math.min(cluster.agents.length, other.agents.length);

      return overlapRatio >= overlapThreshold;
    });

    if (overlapping.length > 0) {
      // Merge all overlapping clusters
      const allSignals = [cluster, ...overlapping].flatMap(c => c.signals);
      const mergedCluster = createClusterFromSignals(allSignals);
      merged.push(mergedCluster);

      used.add(cluster.id);
      overlapping.forEach(c => used.add(c.id));
    } else {
      merged.push(cluster);
      used.add(cluster.id);
    }
  });

  return merged;
}
