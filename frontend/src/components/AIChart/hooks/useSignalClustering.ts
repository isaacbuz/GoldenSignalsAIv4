/**
 * useSignalClustering Hook
 *
 * Manages signal clustering and visualization for agent signals
 */

import { useCallback, useMemo } from 'react';
import { AgentSignal } from '../components/ChartCanvas/types';
import { CoordinateSystem } from '../utils/coordinateSystem';
import {
  clusterSignals,
  filterClustersBySize,
  sortClustersByStrength,
  getClusterColor,
  getClusterImportance,
  SignalCluster
} from '../utils/signalClustering';
import { drawTextWithBackground } from '../utils/canvasSetup';

interface DrawClusteredSignalsParams {
  ctx: CanvasRenderingContext2D;
  signals: AgentSignal[];
  coordinates: CoordinateSystem;
  theme: any;
  minClusterSize?: number;
  showLabels?: boolean;
  animationPhase?: number;
}

export const useSignalClustering = () => {
  /**
   * Draw a single signal cluster
   */
  const drawCluster = useCallback((
    ctx: CanvasRenderingContext2D,
    cluster: SignalCluster,
    coordinates: CoordinateSystem,
    theme: any,
    animationPhase: number = 0
  ) => {
    const x = coordinates.timeToX(cluster.time);
    const y = coordinates.priceToY(cluster.price);
    const viewport = coordinates.getViewportBounds();

    // Skip if outside viewport
    if (x < viewport.x - 50 || x > viewport.x + viewport.width + 50) return;

    ctx.save();

    // Calculate cluster size based on signal count and importance
    const importance = getClusterImportance(cluster);
    const baseSize = 20 + cluster.signals.length * 3;
    const size = baseSize * (0.8 + importance * 0.4);

    // Pulse animation
    const pulseScale = 1 + Math.sin(animationPhase + cluster.time) * 0.1;
    const animatedSize = size * pulseScale;

    // Draw outer glow
    const glowColor = getClusterColor(cluster, theme.palette.mode);
    ctx.shadowBlur = 20 * pulseScale;
    ctx.shadowColor = glowColor;

    // Draw main circle
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, animatedSize);
    gradient.addColorStop(0, glowColor + 'FF');
    gradient.addColorStop(0.6, glowColor + '80');
    gradient.addColorStop(1, glowColor + '20');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, animatedSize, 0, Math.PI * 2);
    ctx.fill();

    // Draw inner circle
    ctx.shadowBlur = 0;
    ctx.fillStyle = theme.palette.background.paper + 'E6';
    ctx.beginPath();
    ctx.arc(x, y, animatedSize * 0.6, 0, Math.PI * 2);
    ctx.fill();

    // Draw border
    ctx.strokeStyle = glowColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, animatedSize, 0, Math.PI * 2);
    ctx.stroke();

    // Draw agent count
    ctx.fillStyle = theme.palette.text.primary;
    ctx.font = `bold ${14 + cluster.signals.length}px Inter, system-ui, -apple-system, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(cluster.signals.length.toString(), x, y - 2);

    // Draw consensus indicator
    if (cluster.consensus !== 'HOLD') {
      const arrow = cluster.consensus === 'BUY' ? '▲' : '▼';
      ctx.font = '16px Inter, system-ui, -apple-system, sans-serif';
      ctx.fillStyle = getClusterColor(cluster, theme.palette.mode);
      ctx.fillText(arrow, x, y + animatedSize + 15);
    }

    // Draw agent names tooltip
    const labelY = cluster.consensus === 'BUY' ? y - animatedSize - 25 : y + animatedSize + 35;
    drawTextWithBackground(
      ctx,
      cluster.agents.slice(0, 3).join(', ') + (cluster.agents.length > 3 ? '...' : ''),
      x,
      labelY,
      {
        font: '11px Inter, system-ui, -apple-system, sans-serif',
        textColor: theme.palette.text.secondary,
        backgroundColor: theme.palette.background.paper + 'CC',
        align: 'center',
        padding: 4,
        borderRadius: 4,
      }
    );

    // Draw strength indicator
    const strengthBarWidth = 40;
    const strengthBarHeight = 4;
    const strengthBarY = labelY + 15;

    ctx.fillStyle = theme.palette.background.paper;
    ctx.fillRect(
      x - strengthBarWidth / 2,
      strengthBarY,
      strengthBarWidth,
      strengthBarHeight
    );

    ctx.fillStyle = getClusterColor(cluster, theme.palette.mode);
    ctx.fillRect(
      x - strengthBarWidth / 2,
      strengthBarY,
      strengthBarWidth * cluster.strength,
      strengthBarHeight
    );

    ctx.restore();
  }, []);

  /**
   * Draw consensus line connecting clusters
   */
  const drawConsensusLine = useCallback((
    ctx: CanvasRenderingContext2D,
    clusters: SignalCluster[],
    coordinates: CoordinateSystem,
    theme: any
  ) => {
    if (clusters.length < 2) return;

    // Sort clusters by time
    const sortedClusters = [...clusters].sort((a, b) => a.time - b.time);

    ctx.save();
    ctx.strokeStyle = theme.palette.mode === 'dark' ? '#FFD70040' : '#FFC10740';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);

    ctx.beginPath();
    sortedClusters.forEach((cluster, i) => {
      const x = coordinates.timeToX(cluster.time);
      const y = coordinates.priceToY(cluster.price);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.restore();
  }, []);

  /**
   * Main function to draw clustered signals
   */
  const drawClusteredSignals = useCallback((params: DrawClusteredSignalsParams) => {
    const {
      ctx,
      signals,
      coordinates,
      theme,
      minClusterSize = 1,
      showLabels = true,
      animationPhase = 0,
    } = params;

    if (!signals || signals.length === 0) return;

    // Cluster the signals
    const clusters = clusterSignals(signals);

    // Filter and sort clusters
    const filteredClusters = filterClustersBySize(clusters, minClusterSize);
    const sortedClusters = sortClustersByStrength(filteredClusters);

    // Draw consensus line first (behind clusters)
    if (sortedClusters.length > 1) {
      drawConsensusLine(ctx, sortedClusters, coordinates, theme);
    }

    // Draw clusters
    sortedClusters.forEach(cluster => {
      drawCluster(ctx, cluster, coordinates, theme, animationPhase);
    });

    // Draw summary statistics
    if (showLabels && sortedClusters.length > 0) {
      const viewport = coordinates.getViewportBounds();
      const totalBuy = sortedClusters.reduce((sum, c) => sum + c.buyCount, 0);
      const totalSell = sortedClusters.reduce((sum, c) => sum + c.sellCount, 0);
      const consensus = totalBuy > totalSell ? 'BULLISH' : totalSell > totalBuy ? 'BEARISH' : 'NEUTRAL';

      drawTextWithBackground(
        ctx,
        `Agent Consensus: ${consensus} (${totalBuy} buy, ${totalSell} sell)`,
        viewport.x + viewport.width - 10,
        viewport.y + 30,
        {
          font: '12px Inter, system-ui, -apple-system, sans-serif',
          textColor: theme.palette.text.primary,
          backgroundColor: theme.palette.background.paper + 'E6',
          align: 'right',
          padding: 6,
          borderRadius: 4,
        }
      );
    }
  }, [drawCluster, drawConsensusLine]);

  /**
   * Get cluster at position (for interactions)
   */
  const getClusterAtPosition = useCallback((
    clusters: SignalCluster[],
    x: number,
    y: number,
    coordinates: CoordinateSystem,
    threshold: number = 30
  ): SignalCluster | null => {
    for (const cluster of clusters) {
      const clusterX = coordinates.timeToX(cluster.time);
      const clusterY = coordinates.priceToY(cluster.price);

      const distance = Math.sqrt(
        Math.pow(x - clusterX, 2) + Math.pow(y - clusterY, 2)
      );

      if (distance <= threshold) {
        return cluster;
      }
    }

    return null;
  }, []);

  return {
    drawClusteredSignals,
    clusterSignals: useMemo(() => clusterSignals, []),
    getClusterAtPosition,
  };
};
