import React, { useState, useEffect } from 'react';
import MainTradingChart from './MainTradingChart';
import SymbolCard from './SymbolCard';
import SentimentGauge from './SentimentGauge';
import { Time } from 'lightweight-charts';
import { apiClient } from '../../services/api';

interface Symbol {
  symbol: string;
  price: number;
  change: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  divergenceCount: number;
  consensus: 'strong' | 'moderate' | 'weak';
}

const HybridDashboard: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [symbols, setSymbols] = useState<Symbol[]>([
    {
      symbol: 'AAPL',
      price: 150.23,
      change: 1.2,
      signal: 'BUY',
      confidence: 78,
      divergenceCount: 2,
      consensus: 'moderate'
    },
    {
      symbol: 'TSLA',
      price: 245.67,
      change: -2.1,
      signal: 'SELL',
      confidence: 65,
      divergenceCount: 0,
      consensus: 'strong'
    },
    {
      symbol: 'NVDA',
      price: 450.12,
      change: 0.3,
      signal: 'HOLD',
      confidence: 52,
      divergenceCount: 3,
      consensus: 'weak'
    }
  ]);

  const [chartData, setChartData] = useState<any[]>([]);
  const [signalData, setSignalData] = useState<any>(null);
  const [marketSentiment, setMarketSentiment] = useState({
    overall: 'bullish' as 'bullish' | 'bearish' | 'neutral',
    confidence: 0.68,
    momentum: 3,
    agentCount: 6
  });

  // WebSocket connection for real-time updates
  // const { data: wsData, isConnected } = useWebSocket('/signals');

  // Fetch chart data
  useEffect(() => {
    fetchChartData(selectedSymbol);
    fetchSignalData(selectedSymbol);
  }, [selectedSymbol]);

  // Process WebSocket data
  useEffect(() => {
    // if (wsData) {
    //   // Update signal data, chart data, etc.
    //   console.log('Received WebSocket data:', wsData);
    // }
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    // if (wsData) {
    //   if (wsData.type === 'signal_update') {
    //     updateSymbolData(wsData.data);
    //   } else if (wsData.type === 'sentiment_update') {
    //     setMarketSentiment(prev => ({ ...prev, ...wsData.data }));
    //   }
    // }
  }, []);

  const fetchChartData = async (symbol: string) => {
    try {
      const data = await apiClient.getHistoricalMarketData(symbol, '1D');
      setChartData(data);
    } catch (error) {
      console.error('Error fetching chart data:', error);
    }
  };

  const fetchSignalData = async (symbol: string) => {
    try {
      const response = await fetch(`/api/v1/hybrid/signals/${symbol}`);
      const data = await response.json();
      setSignalData(data);
    } catch (error) {
      console.error('Error fetching signal data:', error);
    }
  };

  const updateSymbolData = (update: any) => {
    setSymbols(prev => prev.map(sym => 
      sym.symbol === update.symbol 
        ? { ...sym, ...update }
        : sym
    ));
  };

  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  return (
    <div className="min-h-screen bg-background p-4">
      {/* Header */}
      <header className="bg-surface border-b border-surface-light px-6 py-4 mb-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-6">
            <h1 className="text-2xl font-bold text-text-primary">
              🚀 GoldenSignals AI
            </h1>
            <div className="text-text-secondary">
              Portfolio: <span className="text-green-500 font-semibold">$125,430 (+2.3%)</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${/* isConnected */ false ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-text-secondary text-sm">
                {/* isConnected ? 'Live' : 'Connecting...' */}
              </span>
            </div>
            <button className="text-text-secondary hover:text-text-primary">
              ⚙️ Settings
            </button>
          </div>
        </div>
      </header>

      {/* Main Trading Chart */}
      <MainTradingChart
        symbol={selectedSymbol}
        data={[]}
        currentPrice={152.35}
        priceChange={2.45}
        volume={45280000}
        support={148.50}
        resistance={155.80}
        entryPrice={150.25}
        stopLoss={147.80}
        takeProfit={156.50}
        predictionTimeframe="1h"
        predictionConfidence={0.78}
        currentSignal={{ action: 'BUY', confidence: 85 }}
        signals={[
          { time: Date.now() / 1000 - 3600, position: 'belowBar', shape: 'arrowUp', color: '#10B981', text: 'Buy Signal' } as any,
          { time: Date.now() / 1000 - 7200, position: 'aboveBar', shape: 'arrowDown', color: '#EF4444', text: 'Sell Signal' } as any,
        ]}
        divergences={[
          {
            startTime: (Date.now() / 1000 - 10800) as any,
            endTime: (Date.now() / 1000 - 7200) as any,
            startPrice: 149.00,
            endPrice: 151.50,
            type: 'bullish',
            indicator: 'RSI'
          }
        ]}
      />

      {/* Dashboard Grid Below Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* 1. Symbol Watchlist */}
        <div className="bg-surface rounded-lg border border-surface-light p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-4">WATCHLIST</h3>
          <div className="space-y-3">
            {symbols.map(sym => (
              <SymbolCard
                key={sym.symbol}
                {...sym}
                onClick={() => handleSymbolSelect(sym.symbol)}
              />
            ))}
            <button className="w-full py-2 border border-dashed border-surface-light text-text-secondary hover:text-text-primary hover:border-cyan-500 rounded transition-colors">
              + Add Symbol
            </button>
          </div>
        </div>

        {/* 2. Hybrid Signal Analysis */}
        <div className="bg-surface rounded-lg border border-surface-light p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-4">SIGNAL BREAKDOWN</h3>
          {/* <SignalBreakdown
            signals={signalData?.agent_results || []}
            selectedSymbol={selectedSymbol}
            onAgentClick={(agent) => console.log('Agent clicked:', agent)}
          /> */}
          <div className="bg-surface border border-surface-light rounded-lg p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Signal Breakdown</h3>
            <p className="text-text-secondary">Signal breakdown component will show agent matrix here</p>
          </div>
        </div>

        {/* 3. Market Sentiment & Divergences */}
        <div className="bg-surface rounded-lg border border-surface-light p-4 space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-4">MARKET SENTIMENT</h3>
            <SentimentGauge
              sentiment={marketSentiment.overall}
              confidence={marketSentiment.confidence}
              showMomentum={true}
              momentum={marketSentiment.momentum}
            />
          </div>
          
          <div className="border-t border-surface-light pt-4">
            <h3 className="text-lg font-semibold text-text-primary mb-4">DIVERGENCE RADAR</h3>
            {/* <DivergenceRadar
              divergences={divergences}
              onDivergenceClick={(div) => console.log('Divergence clicked:', div)}
            /> */}
            <div className="bg-surface border border-surface-light rounded-lg p-6">
              <h3 className="text-lg font-semibold text-text-primary mb-4">Divergence Radar</h3>
              <p className="text-text-secondary">Divergence detection visualization will appear here</p>
            </div>
          </div>
        </div>

        {/* 4. Performance & Actions */}
        <div className="bg-surface rounded-lg border border-surface-light p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-4">LIVE PERFORMANCE</h3>
          {/* <PerformancePanel
            performance={performanceData}
            recentTrades={[]}
            onTradeAction={(action) => console.log('Trade action:', action)}
          /> */}
          <div className="bg-surface border border-surface-light rounded-lg p-6">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Performance</h3>
            <p className="text-text-secondary">Performance metrics and trade actions will display here</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HybridDashboard; 