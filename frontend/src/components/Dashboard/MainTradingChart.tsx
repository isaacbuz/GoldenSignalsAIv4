import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, Time, UTCTimestamp } from 'lightweight-charts';
import { motion } from 'framer-motion';
import { Volume2, TrendingUp, TrendingDown, Target, ShieldAlert, DollarSign, Activity } from 'lucide-react';

interface ChartData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface SignalData {
  time: Time;
  position: 'aboveBar' | 'belowBar';
  shape: 'arrowUp' | 'arrowDown';
  color: string;
  text: string;
  size?: number;
}

interface DivergenceData {
  startTime: Time;
  endTime: Time;
  startPrice: number;
  endPrice: number;
  type: 'bullish' | 'bearish';
  indicator: string;
}

interface TradeMarker {
  time: Time;
  price: number;
  type: 'entry' | 'exit' | 'stopLoss' | 'takeProfit';
  position?: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square';
  text: string;
}

interface PredictionData {
  time: Time;
  value: number;
}

interface MainTradingChartProps {
  symbol: string;
  data: ChartData[];
  signals?: SignalData[];
  divergences?: DivergenceData[];
  supportResistance?: { price: number; type: 'support' | 'resistance'; strength: number }[];
  patterns?: { type: string; points: { time: Time; price: number }[] }[];
  currentSignal?: { action: 'BUY' | 'SELL' | 'HOLD'; confidence: number };
  showIndicators?: {
    rsi?: boolean;
    macd?: boolean;
    bollingerBands?: boolean;
    volume?: boolean;
  };
  currentPrice?: number;
  priceChange?: number;
  volume?: number;
  support?: number;
  resistance?: number;
  entryPrice?: number;
  exitPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  predictionTimeframe?: '5m' | '15m' | '30m' | '1h' | '4h';
  predictionConfidence?: number;
}

const MainTradingChart: React.FC<MainTradingChartProps> = ({
  symbol,
  data,
  signals = [],
  divergences = [],
  supportResistance = [],
  patterns = [],
  currentSignal,
  showIndicators = { rsi: true, macd: true, bollingerBands: true, volume: true },
  currentPrice = 0,
  priceChange = 0,
  volume = 0,
  support = 0,
  resistance = 0,
  entryPrice,
  exitPrice,
  stopLoss,
  takeProfit,
  predictionTimeframe = '1h',
  predictionConfidence = 0
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candlestickSeries = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeries = useRef<ISeriesApi<"Histogram"> | null>(null);
  const predictionSeries = useRef<ISeriesApi<"Line"> | null>(null);
  const [timeframe, setTimeframe] = useState('1D');
  const [showDivergenceAlert, setShowDivergenceAlert] = useState(false);
  const [showPrediction, setShowPrediction] = useState(true);

  // Generate sample data if none provided
  const generateSampleData = (): ChartData[] => {
    const basePrice = currentPrice || 100;
    const now = new Date();
    const sampleData: ChartData[] = [];

    for (let i = 100; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000);
      const timestamp = Math.floor(time.getTime() / 1000) as UTCTimestamp;
      
      const open = basePrice + (Math.random() - 0.5) * 4;
      const close = open + (Math.random() - 0.5) * 4;
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;
      const volume = Math.floor(Math.random() * 1000000) + 500000;

      sampleData.push({ time: timestamp, open, high, low, close, volume });
    }

    return sampleData;
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    chart.current = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#0F172A' },
        textColor: '#E2E8F0',
      },
      grid: {
        vertLines: { color: '#1E293B' },
        horzLines: { color: '#1E293B' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#475569',
          style: 2,
        },
        horzLine: {
          width: 1,
          color: '#475569',
          style: 2,
        },
      },
      rightPriceScale: {
        borderColor: '#1E293B',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: '#1E293B',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Add candlestick series
    candlestickSeries.current = chart.current.addCandlestickSeries({
      upColor: '#10B981',
      downColor: '#EF4444',
      borderUpColor: '#10B981',
      borderDownColor: '#EF4444',
      wickUpColor: '#10B981',
      wickDownColor: '#EF4444',
    });

    // Use sample data if no data provided
    const chartData = data.length > 0 ? data : generateSampleData();
    candlestickSeries.current.setData(chartData);

    // Add volume series
    if (showIndicators.volume) {
      volumeSeries.current = chart.current.addHistogramSeries({
        color: '#3B82F6',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
      });

      const volumeData = chartData.map(d => ({
        time: d.time,
        value: d.volume,
        color: d.close > d.open ? '#10B98180' : '#EF444480',
      }));

      volumeSeries.current.setData(volumeData);
    }

    // Add prediction line series
    predictionSeries.current = chart.current.addLineSeries({
      color: '#8B5CF6',
      lineWidth: 2,
      lineStyle: 2, // Dashed line
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: '#8B5CF6',
      crosshairMarkerBackgroundColor: '#8B5CF6',
    });

    // Add support/resistance lines
    supportResistance.forEach(level => {
      const lineSeries = chart.current!.addLineSeries({
        color: level.type === 'support' ? '#10B981' : '#EF4444',
        lineWidth: level.strength > 0.8 ? 2 : 1,
        lineStyle: level.strength > 0.8 ? 0 : 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      const lineData = chartData.map(d => ({
        time: d.time,
        value: level.price,
      }));

      lineSeries.setData(lineData);
    });

    // Add Bollinger Bands if enabled
    if (showIndicators.bollingerBands) {
      const upperBand = chart.current.addLineSeries({
        color: '#8B5CF6',
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      const lowerBand = chart.current.addLineSeries({
        color: '#8B5CF6',
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      // Calculate Bollinger Bands (simplified)
      const period = 20;
      const stdDev = 2;
      const bandsData = calculateBollingerBands(chartData, period, stdDev);
      
      upperBand.setData(bandsData.upper);
      lowerBand.setData(bandsData.lower);
    }

    // Add pattern overlays
    patterns.forEach(pattern => {
      const patternSeries = chart.current!.addLineSeries({
        color: '#06B6D4',
        lineWidth: 2,
        lineStyle: 0,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      const patternData = pattern.points.map(p => ({
        time: p.time,
        value: p.price,
      }));

      patternSeries.setData(patternData);
    });

    // Add signal markers
    if (candlestickSeries.current && signals.length > 0) {
      candlestickSeries.current.setMarkers(signals.map(signal => ({
        time: signal.time,
        position: signal.position,
        shape: signal.shape,
        color: signal.color,
        text: signal.text,
      })));
    }

    // Add divergence lines
    divergences.forEach(div => {
      const divergenceSeries = chart.current!.addLineSeries({
        color: div.type === 'bullish' ? '#10B981' : '#EF4444',
        lineWidth: 2,
        lineStyle: 3,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      divergenceSeries.setData([
        { time: div.startTime, value: div.startPrice },
        { time: div.endTime, value: div.endPrice },
      ]);
    });

    // Add trading markers
    addTradingMarkers(chartData);

    // Add prediction trendline
    if (showPrediction) {
      addPredictionTrendline(chartData);
    }

    // Add stop loss and take profit lines
    addTradingLevels();

    // Check for divergences to show alert
    if (divergences.length > 0) {
      setShowDivergenceAlert(true);
      setTimeout(() => setShowDivergenceAlert(false), 5000);
    }

    // Handle resize
    const handleResize = () => {
      if (chart.current && chartContainerRef.current) {
        chart.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chart.current) {
        chart.current.remove();
      }
    };
  }, [data, signals, divergences, supportResistance, patterns, showIndicators, showPrediction]);

  const calculateBollingerBands = (data: ChartData[], period: number, stdDev: number) => {
    const upper: { time: Time; value: number }[] = [];
    const lower: { time: Time; value: number }[] = [];

    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const avg = slice.reduce((sum, d) => sum + d.close, 0) / period;
      const variance = slice.reduce((sum, d) => sum + Math.pow(d.close - avg, 2), 0) / period;
      const std = Math.sqrt(variance);

      upper.push({ time: data[i].time, value: avg + stdDev * std });
      lower.push({ time: data[i].time, value: avg - stdDev * std });
    }

    return { upper, lower };
  };

  const addTradingMarkers = (chartData: ChartData[]) => {
    if (!candlestickSeries.current || chartData.length === 0) return;

    const markers: any[] = [];
    const lastTime = chartData[chartData.length - 1].time;
    const currentTimestamp = typeof lastTime === 'number' ? lastTime : Date.now() / 1000;

    // Entry marker
    if (entryPrice) {
      const entryTime = (currentTimestamp - 7200) as Time; // 2 hours ago
      markers.push({
        time: entryTime,
        position: 'belowBar',
        color: '#10B981',
        shape: 'arrowUp',
        text: `Entry: $${entryPrice}`,
      });
    }

    // Exit marker
    if (exitPrice) {
      const exitTime = (currentTimestamp - 1800) as Time; // 30 min ago
      markers.push({
        time: exitTime,
        position: 'aboveBar',
        color: '#3B82F6',
        shape: 'arrowDown',
        text: `Exit: $${exitPrice}`,
      });
    }

    // Add trading signal markers based on current signal
    if (currentSignal) {
      const signalTime = currentTimestamp as Time;
      if (currentSignal.action === 'BUY') {
        markers.push({
          time: signalTime,
          position: 'belowBar',
          color: '#10B981',
          shape: 'arrowUp',
          text: `Buy Signal (${currentSignal.confidence}%)`,
        });
      } else if (currentSignal.action === 'SELL') {
        markers.push({
          time: signalTime,
          position: 'aboveBar',
          color: '#EF4444',
          shape: 'arrowDown',
          text: `Sell Signal (${currentSignal.confidence}%)`,
        });
      }
    }

    if (markers.length > 0) {
      candlestickSeries.current.setMarkers(markers);
    }
  };

  const addPredictionTrendline = (historicalData: ChartData[]) => {
    if (!predictionSeries.current || !chart.current || historicalData.length === 0) return;

    const lastCandle = historicalData[historicalData.length - 1];
    if (!lastCandle) return;

    const currentTime = lastCandle.time;
    const currentPrice = lastCandle.close;
    const currentTimestamp = typeof currentTime === 'number' ? currentTime : Date.now() / 1000;

    // Calculate prediction based on recent trend and ML signals
    const recentData = historicalData.slice(-20); // Last 20 candles
    const trend = calculateTrend(recentData);
    
    // Generate prediction points
    const predictionPoints: { time: Time; value: number }[] = [];
    const timeframes = {
      '5m': 5 * 60,
      '15m': 15 * 60,
      '30m': 30 * 60,
      '1h': 60 * 60,
      '4h': 4 * 60 * 60,
    };
    
    const predictionLength = timeframes[predictionTimeframe];
    const intervals = 10; // Number of points in prediction
    const intervalTime = predictionLength / intervals;

    // Add current point as start of prediction
    predictionPoints.push({
      time: currentTime,
      value: currentPrice,
    });

    // Generate prediction curve
    let predictedPrice = currentPrice;
    for (let i = 1; i <= intervals; i++) {
      const futureTime = (currentTimestamp + (intervalTime * i)) as Time;
      
      // Apply trend with some volatility
      const trendFactor = trend * (0.002 * i); // 0.2% per interval in trend direction
      const volatility = (Math.random() - 0.5) * 0.001 * currentPrice; // 0.1% random volatility
      predictedPrice = predictedPrice * (1 + trendFactor) + volatility;

      predictionPoints.push({
        time: futureTime,
        value: predictedPrice,
      });
    }

    predictionSeries.current.setData(predictionPoints);

    // Update chart time scale to show prediction
    const visibleLogicalRange = chart.current.timeScale().getVisibleLogicalRange();
    if (visibleLogicalRange) {
      chart.current.timeScale().setVisibleRange({
        from: historicalData[Math.max(0, historicalData.length - 50)].time,
        to: predictionPoints[predictionPoints.length - 1].time,
      });
    }
  };

  const calculateTrend = (data: ChartData[]): number => {
    if (data.length < 2) return 0;
    
    const firstPrice = data[0].close;
    const lastPrice = data[data.length - 1].close;
    const priceChange = (lastPrice - firstPrice) / firstPrice;
    
    // Calculate momentum
    let momentum = 0;
    for (let i = 1; i < data.length; i++) {
      momentum += (data[i].close - data[i - 1].close) / data[i - 1].close;
    }
    momentum = momentum / data.length;
    
    // Combine price change and momentum
    return priceChange * 0.7 + momentum * 0.3;
  };

  const addTradingLevels = () => {
    if (!candlestickSeries.current) return;

    // Stop Loss line
    if (stopLoss && stopLoss > 0) {
      candlestickSeries.current.createPriceLine({
        price: stopLoss,
        color: '#DC2626',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'Stop Loss',
      });
    }

    // Take Profit line
    if (takeProfit && takeProfit > 0) {
      candlestickSeries.current.createPriceLine({
        price: takeProfit,
        color: '#059669',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'Take Profit',
      });
    }

    // Current price line
    if (currentPrice > 0) {
      candlestickSeries.current.createPriceLine({
        price: currentPrice,
        color: '#3B82F6',
        lineWidth: 1,
        lineStyle: 0, // Solid
        axisLabelVisible: true,
        title: 'Current',
      });
    }
  };

  const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

  return (
    <div className="relative bg-background rounded-lg border border-surface-light p-4">
      {/* Header */}
      <div className="flex justify-between items-start mb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h2 className="text-3xl font-bold text-text-primary">{symbol}</h2>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              priceChange >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <DollarSign className="w-4 h-4 text-gray-400" />
              <span className="text-white font-medium">${currentPrice.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-gray-400" />
              <span className="text-gray-300">{(volume / 1000000).toFixed(2)}M</span>
            </div>
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-purple-400" />
              <span className="text-purple-300">
                Prediction: {predictionTimeframe} • {(predictionConfidence * 100).toFixed(0)}% confidence
              </span>
            </div>
          </div>
        </div>

        {/* Timeframe Selector */}
        <div className="flex items-center gap-2">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  timeframe === tf
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
          <button
            onClick={() => setShowPrediction(!showPrediction)}
            className={`px-3 py-1 text-sm rounded-md transition-colors ${
              showPrediction
                ? 'bg-purple-600/20 text-purple-400 border border-purple-500'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            {showPrediction ? '🔮 Prediction ON' : '🔮 Prediction OFF'}
          </button>
        </div>
      </div>

      {/* Trading Levels Display */}
      <div className="flex gap-4 mb-4 text-sm">
        {entryPrice && (
          <div className="flex items-center gap-2 bg-green-500/10 px-3 py-1 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-green-400">Entry: ${entryPrice}</span>
          </div>
        )}
        {stopLoss && (
          <div className="flex items-center gap-2 bg-red-500/10 px-3 py-1 rounded-lg">
            <ShieldAlert className="w-4 h-4 text-red-400" />
            <span className="text-red-400">SL: ${stopLoss}</span>
          </div>
        )}
        {takeProfit && (
          <div className="flex items-center gap-2 bg-green-500/10 px-3 py-1 rounded-lg">
            <Target className="w-4 h-4 text-green-400" />
            <span className="text-green-400">TP: ${takeProfit}</span>
          </div>
        )}
        {exitPrice && (
          <div className="flex items-center gap-2 bg-blue-500/10 px-3 py-1 rounded-lg">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="text-blue-400">Exit: ${exitPrice}</span>
          </div>
        )}
      </div>

      {/* Chart Container */}
      <div ref={chartContainerRef} className="w-full h-[500px] bg-gray-950 rounded-lg" />

      {/* Divergence Alert */}
      {showDivergenceAlert && divergences.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="absolute top-20 right-4 bg-yellow-500/20 border border-yellow-500 rounded-lg p-4 shadow-lg"
        >
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-yellow-500" />
            <span className="text-yellow-500 font-semibold">
              {divergences[0].type === 'bullish' ? 'Bullish' : 'Bearish'} Divergence Detected!
            </span>
          </div>
          <p className="text-sm text-text-secondary mt-1">
            {divergences[0].indicator} showing {divergences[0].type} divergence
          </p>
        </motion.div>
      )}

      {/* Pattern Detection Overlay */}
      {patterns.length > 0 && (
        <div className="absolute top-4 left-4 bg-background/90 rounded-lg p-3 shadow-lg">
          <h4 className="text-sm font-semibold text-text-primary mb-2">Patterns Detected:</h4>
          {patterns.map((pattern, idx) => (
            <div key={idx} className="text-xs text-cyan-400 mb-1">
              • {pattern.type}
            </div>
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="flex justify-between items-center mt-4 text-xs text-gray-400">
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-green-500"></div>
            <span>Support: ${support.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-red-500"></div>
            <span>Resistance: ${resistance.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-purple-500 opacity-50" style={{ borderStyle: 'dashed', borderWidth: '1px 0 0 0' }}></div>
            <span>AI Prediction</span>
          </div>
        </div>
        {divergences.length > 0 && (
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-yellow-400" />
            <span className="text-yellow-400">{divergences.length} Divergence{divergences.length > 1 ? 's' : ''} Detected</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default MainTradingChart; 