/**
 * Market data hook
 * Fetches and manages real-time market data
 */

import { useState, useEffect, useRef } from 'react';
import { CandlestickData, Time } from 'lightweight-charts';

interface MarketData {
  candles: CandlestickData[];
  volumes?: { time: Time; value: number; color: string }[];
}

interface UseMarketDataReturn {
  data: MarketData | null;
  price: number;
  change: number;
  changePercent: number;
  isLive: boolean;
  error: string | null;
  loading: boolean;
}

export function useMarketData(symbol: string, timeframe: string): UseMarketDataReturn {
  const [data, setData] = useState<MarketData | null>(null);
  const [price, setPrice] = useState(0);
  const [change, setChange] = useState(0);
  const [changePercent, setChangePercent] = useState(0);
  const [isLive, setIsLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);

  // Timeframe mapping
  const periodMap: Record<string, string> = {
    '1D': '1d',
    '5D': '5d',
    '1M': '1mo',
    '3M': '3mo',
    '1Y': '1y',
    'ALL': '2y',
  };

  const intervalMap: Record<string, string> = {
    '1D': '5m',
    '5D': '30m',
    '1M': '1d',
    '3M': '1d',
    '1Y': '1wk',
    'ALL': '1mo',
  };

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `http://localhost:8000/api/v1/market-data/${symbol}/history?period=${periodMap[timeframe]}&interval=${intervalMap[timeframe]}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch market data');
        }

        const result = await response.json();
        const apiData = result.data || [];

        if (apiData.length === 0) {
          throw new Error('No data available');
        }

        // Process candles
        const candles: CandlestickData[] = apiData.map((d: any) => ({
          time: d.time as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));

        // Process volumes
        const volumes = apiData.map((d: any, i: number) => ({
          time: d.time as Time,
          value: d.volume || 0,
          color: candles[i].close >= candles[i].open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
        }));

        setData({ candles, volumes });

        // Update price info
        const lastCandle = candles[candles.length - 1];
        const firstCandle = candles[0];
        setPrice(lastCandle.close);
        setChange(lastCandle.close - firstCandle.open);
        setChangePercent(((lastCandle.close - firstCandle.open) / firstCandle.open) * 100);

      } catch (err) {
        console.error('Error fetching market data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Set up refresh interval
    const refreshInterval = timeframe === '1D' ? 10000 :
                           timeframe === '5D' ? 30000 :
                           60000;

    const intervalId = setInterval(fetchData, refreshInterval);

    return () => clearInterval(intervalId);
  }, [symbol, timeframe]);

  // WebSocket connection
  useEffect(() => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Connect to WebSocket
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(`ws://localhost:8000/ws/v2/signals/${symbol}`);

        ws.onopen = () => {
          console.log('WebSocket connected for', symbol);
          setIsLive(true);
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            // Update price in real-time
            if (message.type === 'price_update' || message.price) {
              const newPrice = message.price || message.data?.price;
              if (newPrice && data) {
                setPrice(newPrice);

                // Update the last candle
                const updatedCandles = [...data.candles];
                const lastCandle = updatedCandles[updatedCandles.length - 1];
                updatedCandles[updatedCandles.length - 1] = {
                  ...lastCandle,
                  close: newPrice,
                  high: Math.max(lastCandle.high, newPrice),
                  low: Math.min(lastCandle.low, newPrice),
                };

                setData({ ...data, candles: updatedCandles });
              }
            }
          } catch (error) {
            console.error('WebSocket message error:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsLive(false);
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');
          setIsLive(false);
          // Reconnect after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };

        wsRef.current = ws;
      } catch (error) {
        console.error('WebSocket connection error:', error);
        setIsLive(false);
      }
    };

    // Connect after a short delay
    const timeout = setTimeout(connectWebSocket, 1000);

    return () => {
      clearTimeout(timeout);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [symbol, data]);

  return {
    data,
    price,
    change,
    changePercent,
    isLive,
    error,
    loading,
  };
}
