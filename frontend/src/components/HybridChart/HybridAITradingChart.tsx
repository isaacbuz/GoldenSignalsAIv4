import React, { useRef, useEffect, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, Time, MouseEventParams } from 'lightweight-charts';
import * as d3 from 'd3';
import { Box, useTheme, alpha } from '@mui/material';
import { useRealtimeChart } from '../../hooks/useRealtimeChart';
import { aiPredictionService } from '../../services/aiPredictionService';
import { llmAdvisor } from '../../services/llmAdvisor';
import { ChartDataPoint, Pattern, Signal, PredictionPoint } from '../../types/chart';
import { debounce } from 'lodash';
import logger from '../../services/logger';


interface HybridAITradingChartProps {
  symbol: string;
  timeframe: string;
  onSignalReceived?: (signal: Signal) => void;
  height?: number;
}

interface TimeRange {
  from: number;
  to: number;
}

export const HybridAITradingChart: React.FC<HybridAITradingChartProps> = ({
  symbol,
  timeframe,
  onSignalReceived,
  height = 600
}) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const [dimensions, setDimensions] = useState({ width: 800, height });
  const [aiPredictions, setAIPredictions] = useState<PredictionPoint[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [tooltipData, setTooltipData] = useState<any>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Initialize Lightweight Charts
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: dimensions.width,
      height: dimensions.height,
      layout: {
        background: { type: 'solid', color: 'transparent' },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: {
          color: alpha(theme.palette.divider, 0.3),
          style: 2, // Dashed
          visible: true
        },
        horzLines: {
          color: alpha(theme.palette.divider, 0.3),
          style: 2,
          visible: true
        },
      },
      crosshair: {
        mode: 0, // Normal mode
        vertLine: {
          color: theme.palette.primary.main,
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
        },
        horzLine: {
          color: theme.palette.primary.main,
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
        },
      },
      timeScale: {
        borderColor: theme.palette.divider,
        timeVisible: true,
        secondsVisible: timeframe === '1m' || timeframe === '5m',
      },
      rightPriceScale: {
        borderColor: theme.palette.divider,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2, // Space for volume
        },
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00FF88',
      downColor: '#FF4444',
      borderUpColor: '#00FF88',
      borderDownColor: '#FF4444',
      wickUpColor: '#00FF88',
      wickDownColor: '#FF4444',
    });

    candleSeriesRef.current = candleSeries;

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // Use default scale
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    volumeSeriesRef.current = volumeSeries;

    // Position overlays
    positionOverlays();

    // Subscribe to chart events
    chart.subscribeCrosshairMove(handleCrosshairMove);
    chart.subscribeClick(handleChartClick);
    chart.timeScale().subscribeVisibleTimeRangeChange(debouncedSyncOverlays);

    return () => {
      chart.remove();
    };
  }, [theme, dimensions]);

  // Position overlay canvases
  const positionOverlays = () => {
    if (!containerRef.current || !canvasRef.current || !svgRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();

    // Canvas overlay
    canvasRef.current.width = rect.width;
    canvasRef.current.height = rect.height;
    canvasRef.current.style.position = 'absolute';
    canvasRef.current.style.top = '0';
    canvasRef.current.style.left = '0';
    canvasRef.current.style.pointerEvents = 'none';

    // SVG overlay
    svgRef.current.style.position = 'absolute';
    svgRef.current.style.top = '0';
    svgRef.current.style.left = '0';
    svgRef.current.style.width = `${rect.width}px`;
    svgRef.current.style.height = `${rect.height}px`;
  };

  // Coordinate conversion helpers
  const convertToPixels = useCallback((time: Time, price: number): { x: number; y: number } | null => {
    if (!chartRef.current || !candleSeriesRef.current) return null;

    const timeScale = chartRef.current.timeScale();
    const priceScale = candleSeriesRef.current.priceScale();

    const x = timeScale.timeToCoordinate(time);
    const y = priceScale.priceToCoordinate(price);

    if (x === null || y === null) return null;

    return { x, y };
  }, []);

  // Sync overlays with chart movement
  const syncOverlays = useCallback(() => {
    if (!chartRef.current || !canvasRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Get visible range
    const timeScale = chartRef.current.timeScale();
    const visibleRange = timeScale.getVisibleRange();

    if (!visibleRange) return;

    // Draw AI elements
    drawAIPredictions(ctx, visibleRange);
    drawPatterns(ctx, visibleRange);
    drawSignals(ctx, visibleRange);

    // Update D3 elements
    updateD3Elements();
  }, [aiPredictions, patterns, signals]);

  const debouncedSyncOverlays = useCallback(debounce(syncOverlays, 16), [syncOverlays]);

  // Draw AI predictions on canvas
  const drawAIPredictions = (ctx: CanvasRenderingContext2D, range: TimeRange) => {
    const visiblePredictions = aiPredictions.filter(p =>
      p.time >= range.from && p.time <= range.to
    );

    if (visiblePredictions.length < 2) return;

    ctx.save();

    // Main prediction line
    ctx.strokeStyle = '#FFD700';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.shadowColor = '#FFD700';
    ctx.shadowBlur = 10;

    ctx.beginPath();
    visiblePredictions.forEach((pred, i) => {
      const point = convertToPixels(pred.time as Time, pred.price);
      if (!point) return;

      if (i === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();

    // Confidence bounds
    if (visiblePredictions[0].upperBound && visiblePredictions[0].lowerBound) {
      ctx.fillStyle = 'rgba(255, 215, 0, 0.1)';
      ctx.beginPath();

      // Upper bound
      visiblePredictions.forEach((pred, i) => {
        const point = convertToPixels(pred.time as Time, pred.upperBound!);
        if (!point) return;

        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });

      // Lower bound (reverse)
      for (let i = visiblePredictions.length - 1; i >= 0; i--) {
        const pred = visiblePredictions[i];
        const point = convertToPixels(pred.time as Time, pred.lowerBound!);
        if (point) ctx.lineTo(point.x, point.y);
      }

      ctx.closePath();
      ctx.fill();
    }

    ctx.restore();
  };

  // Draw patterns
  const drawPatterns = (ctx: CanvasRenderingContext2D, range: TimeRange) => {
    patterns.forEach(pattern => {
      if (!pattern.points.some(p => p.time >= range.from && p.time <= range.to)) return;

      ctx.save();

      // Pattern outline
      ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);

      ctx.beginPath();
      pattern.points.forEach((p, i) => {
        const point = convertToPixels(p.time as Time, p.price);
        if (!point) return;

        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
      ctx.stroke();

      // Pattern fill
      ctx.fillStyle = 'rgba(255, 215, 0, 0.1)';
      ctx.fill();

      // Pattern label
      if (pattern.points.length > 0) {
        const midIndex = Math.floor(pattern.points.length / 2);
        const midPoint = pattern.points[midIndex];
        const labelPoint = convertToPixels(midPoint.time as Time, midPoint.price);

        if (labelPoint) {
          ctx.fillStyle = theme.palette.text.primary;
          ctx.font = '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(pattern.type, labelPoint.x, labelPoint.y - 10);
        }
      }

      ctx.restore();
    });
  };

  // Draw signals
  const drawSignals = (ctx: CanvasRenderingContext2D, range: TimeRange) => {
    signals.forEach(signal => {
      if (signal.time < range.from || signal.time > range.to) return;

      const point = convertToPixels(signal.time as Time, signal.price);
      if (!point) return;

      ctx.save();

      // Signal arrow
      const arrowSize = 12;
      const offset = signal.action === 'BUY' ? 20 : -20;

      ctx.fillStyle = signal.action === 'BUY' ? '#00FF88' : '#FF4444';
      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = 10;

      ctx.beginPath();
      if (signal.action === 'BUY') {
        // Up arrow
        ctx.moveTo(point.x, point.y + offset);
        ctx.lineTo(point.x - arrowSize/2, point.y + offset + arrowSize);
        ctx.lineTo(point.x + arrowSize/2, point.y + offset + arrowSize);
      } else {
        // Down arrow
        ctx.moveTo(point.x, point.y + offset);
        ctx.lineTo(point.x - arrowSize/2, point.y + offset - arrowSize);
        ctx.lineTo(point.x + arrowSize/2, point.y + offset - arrowSize);
      }
      ctx.closePath();
      ctx.fill();

      // Confidence ring
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 2;
      ctx.globalAlpha = signal.confidence;
      ctx.beginPath();
      ctx.arc(point.x, point.y + offset, arrowSize + 5, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
    });
  };

  // Update D3 interactive elements
  const updateD3Elements = useCallback(() => {
    if (!svgRef.current || !chartRef.current) return;

    const svg = d3.select(svgRef.current);

    // Clear existing elements
    svg.selectAll('*').remove();

    // Add filter definitions
    const defs = svg.append('defs');
    const filter = defs.append('filter')
      .attr('id', 'glow');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', '4')
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode')
      .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
      .attr('in', 'SourceGraphic');

    // Add interactive annotations
    signals.forEach(signal => {
      if (signal.reasoning) {
        const point = convertToPixels(signal.time as Time, signal.price);
        if (!point) return;

        const g = svg.append('g')
          .attr('class', 'annotation')
          .style('cursor', 'pointer');

        g.append('circle')
          .attr('cx', point.x)
          .attr('cy', point.y)
          .attr('r', 5)
          .style('fill', signal.action === 'BUY' ? '#00FF88' : '#FF4444')
          .style('filter', 'url(#glow)')
          .on('mouseover', () => {
            setTooltipData({
              signal,
              reasoning: signal.reasoning,
              confidence: signal.confidence,
            });
            setTooltipPosition({ x: point.x, y: point.y });
          })
          .on('mouseout', () => {
            setTooltipData(null);
          });
      }
    });
  }, [signals, convertToPixels]);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        const { period, interval } = mapTimeframeToPeriodInterval(timeframe);
        const response = await fetch(
          `http://localhost:8000/api/v1/market-data/${symbol}/history?period=${period}&interval=${interval}`
        );
        const data = await response.json();

        if (data.data && candleSeriesRef.current && volumeSeriesRef.current) {
          const candleData = data.data.map((d: any) => ({
            time: d.time as Time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
          }));

          const volumeData = data.data.map((d: any) => ({
            time: d.time as Time,
            value: d.volume,
            color: d.close >= d.open ? '#00FF88' : '#FF4444',
          }));

          candleSeriesRef.current.setData(candleData);
          volumeSeriesRef.current.setData(volumeData);
        }
      } catch (error) {
        logger.error('Failed to load data:', error);
      }
    };

    loadData();
  }, [symbol, timeframe]);

  // Real-time updates
  useEffect(() => {
    let intervalId: number;

    const updateData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/market-data/${symbol}`);
        const data = await response.json();

        if (data && candleSeriesRef.current) {
          const now = Math.floor(Date.now() / 1000);
          candleSeriesRef.current.update({
            time: now as Time,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.price,
          });

          // Trigger AI analysis periodically
          if (shouldRunAnalysis()) {
            runAIAnalysis();
          }
        }
      } catch (error) {
        logger.error('Update failed:', error);
      }
    };

    // Update based on timeframe
    const updateInterval = getUpdateInterval(timeframe);
    intervalId = window.setInterval(updateData, updateInterval);

    return () => clearInterval(intervalId);
  }, [symbol, timeframe]);

  // AI Analysis
  const runAIAnalysis = async () => {
    if (isAnalyzing) return;

    setIsAnalyzing(true);
    try {
      const analysis = await aiPredictionService.analyzeSymbol(symbol, timeframe);

      if (analysis.predictions) {
        setAIPredictions(analysis.predictions);
      }

      if (analysis.patterns) {
        setPatterns(analysis.patterns);
      }

      // Get LLM advice
      if (analysis.currentPrice) {
        const advice = await llmAdvisor.getTradeAdvice({
          symbol,
          price: analysis.currentPrice,
          indicators: analysis.indicators,
          patterns: analysis.patterns,
        });

        if (advice && advice.action !== 'HOLD') {
          const newSignal: Signal = {
            id: Date.now().toString(),
            symbol,
            time: Math.floor(Date.now() / 1000),
            price: analysis.currentPrice,
            action: advice.action,
            confidence: advice.confidence,
            reasoning: advice.reasoning,
            target_price: advice.take_profits?.[0],
            stop_loss: advice.stop_loss,
          };

          setSignals(prev => [...prev, newSignal]);
          onSignalReceived?.(newSignal);
        }
      }

      // Sync overlays after analysis
      requestAnimationFrame(syncOverlays);
    } catch (error) {
      logger.error('AI analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Event handlers
  const handleCrosshairMove = (param: MouseEventParams) => {
    // Update crosshair data for tooltips
  };

  const handleChartClick = (param: MouseEventParams) => {
    // Handle clicks on chart elements
  };

  // Helper functions
  const shouldRunAnalysis = () => {
    // Run analysis every 5 minutes for lower timeframes
    const now = Date.now();
    const lastAnalysis = parseInt(localStorage.getItem(`lastAnalysis_${symbol}`) || '0');
    const interval = timeframe === '1m' || timeframe === '5m' ? 300000 : 900000; // 5 or 15 minutes

    if (now - lastAnalysis > interval) {
      localStorage.setItem(`lastAnalysis_${symbol}`, now.toString());
      return true;
    }

    return false;
  };

  const getUpdateInterval = (tf: string) => {
    const intervals: Record<string, number> = {
      '1m': 1000,
      '5m': 5000,
      '15m': 15000,
      '30m': 30000,
      '1h': 60000,
      '4h': 240000,
      '1d': 600000,
    };

    return intervals[tf] || 60000;
  };

  const mapTimeframeToPeriodInterval = (tf: string) => {
    const mapping: Record<string, { period: string; interval: string }> = {
      '1m': { period: '1d', interval: '1m' },
      '5m': { period: '5d', interval: '5m' },
      '15m': { period: '5d', interval: '15m' },
      '30m': { period: '5d', interval: '30m' },
      '1h': { period: '1mo', interval: '1h' },
      '4h': { period: '3mo', interval: '1d' },
      '1d': { period: '1y', interval: '1d' },
    };

    return mapping[tf] || { period: '1mo', interval: '1d' };
  };

  // Resize handling
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        const { width } = containerRef.current.getBoundingClientRect();
        setDimensions(prev => ({ ...prev, width }));
        chartRef.current.resize(width, height);
        positionOverlays();
        requestAnimationFrame(syncOverlays);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Initial sizing

    return () => window.removeEventListener('resize', handleResize);
  }, [height]);

  return (
    <Box position="relative" width="100%" height={height}>
      {/* Lightweight Charts container */}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Canvas overlay for AI drawings */}
      <canvas ref={canvasRef} />

      {/* SVG overlay for interactive elements */}
      <svg ref={svgRef} />

      {/* Tooltip */}
      {tooltipData && (
        <Box
          position="absolute"
          left={tooltipPosition.x}
          top={tooltipPosition.y - 100}
          bgcolor="background.paper"
          border={1}
          borderColor="divider"
          borderRadius={1}
          p={1}
          sx={{
            transform: 'translateX(-50%)',
            pointerEvents: 'none',
            zIndex: 1000,
            minWidth: 200,
          }}
        >
          <Box fontSize="0.875rem" fontWeight="bold" mb={0.5}>
            AI Signal: {tooltipData.signal.action}
          </Box>
          <Box fontSize="0.75rem" color="text.secondary">
            Confidence: {(tooltipData.confidence * 100).toFixed(0)}%
          </Box>
          <Box fontSize="0.75rem" mt={0.5}>
            {tooltipData.reasoning}
          </Box>
        </Box>
      )}
    </Box>
  );
};
