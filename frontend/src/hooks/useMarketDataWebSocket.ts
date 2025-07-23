import { useEffect, useState, useCallback, useRef } from 'react';
import { getMarketDataWebSocket, MarketDataUpdate } from '../services/websocket/marketDataWebSocket';
import logger from '../services/logger';


interface UseMarketDataWebSocketOptions {
  symbol: string;
  enabled?: boolean;
  onUpdate?: (update: MarketDataUpdate) => void;
}

interface UseMarketDataWebSocketReturn {
  latestPrice: number | null;
  latestVolume: number | null;
  lastUpdate: Date | null;
  isConnected: boolean;
  priceHistory: Array<{ time: number; price: number }>;
}

export const useMarketDataWebSocket = ({
  symbol,
  enabled = true,
  onUpdate
}: UseMarketDataWebSocketOptions): UseMarketDataWebSocketReturn => {
  const [latestPrice, setLatestPrice] = useState<number | null>(null);
  const [latestVolume, setLatestVolume] = useState<number | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [priceHistory, setPriceHistory] = useState<Array<{ time: number; price: number }>>([]);
  const priceHistoryRef = useRef<Array<{ time: number; price: number }>>([]);

  // Get Finnhub API key from environment
  const finnhubApiKey = import.meta.env.VITE_FINNHUB_API_KEY || 'd0ihu29r01qrfsag9qo0d0ihu29r01qrfsag9qog';

  const handleUpdate = useCallback((update: MarketDataUpdate) => {
    // Update latest values
    setLatestPrice(update.price);
    setLatestVolume(update.volume);
    setLastUpdate(new Date(update.timestamp));

    // Add to price history (keep last 100 points)
    const newPoint = { time: update.timestamp, price: update.price };
    priceHistoryRef.current = [...priceHistoryRef.current.slice(-99), newPoint];
    setPriceHistory(priceHistoryRef.current);

    // Call custom handler if provided
    if (onUpdate) {
      onUpdate(update);
    }
  }, [onUpdate]);

  useEffect(() => {
    if (!enabled || !symbol) {
      return;
    }

    let unsubscribe: (() => void) | null = null;
    let connectionCheckInterval: NodeJS.Timeout | null = null;

    try {
      const ws = getMarketDataWebSocket(finnhubApiKey);

      // Subscribe to symbol
      unsubscribe = ws.subscribe(symbol, handleUpdate);

      // Check connection status periodically
      connectionCheckInterval = setInterval(() => {
        setIsConnected(ws.isConnected());
      }, 1000);

      // Initial connection check
      setIsConnected(ws.isConnected());

    } catch (error) {
      logger.error('Failed to connect to market data WebSocket:', error);
    }

    // Cleanup
    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
      if (connectionCheckInterval) {
        clearInterval(connectionCheckInterval);
      }
    };
  }, [symbol, enabled, finnhubApiKey, handleUpdate]);

  return {
    latestPrice,
    latestVolume,
    lastUpdate,
    isConnected,
    priceHistory
  };
};
