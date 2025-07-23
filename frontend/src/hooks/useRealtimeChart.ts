/**
 * Real-time chart hook for smooth price updates without full redraws
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { backendMarketDataService } from '../services/backendMarketDataService';
import logger from '../services/logger';


interface RealtimePrice {
  symbol: string;
  price: number;
  time: number;
  volume?: number;
  bid?: number;
  ask?: number;
}

interface UseRealtimeChartOptions {
  symbol: string;
  onPriceUpdate?: (price: RealtimePrice) => void;
  updateInterval?: number;
}

export const useRealtimeChart = ({
  symbol,
  onPriceUpdate,
  updateInterval = 1000 // 1 second updates
}: UseRealtimeChartOptions) => {
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const priceBufferRef = useRef<RealtimePrice[]>([]);
  const lastUpdateRef = useRef<number>(0);
  const updateTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef<boolean>(true);

  // Smooth price update with animation
  const updatePrice = useCallback((newPrice: RealtimePrice) => {
    if (!isMountedRef.current) return;

    const now = Date.now();

    // Buffer prices to prevent too frequent updates
    priceBufferRef.current.push(newPrice);

    // Limit buffer size to prevent memory leak
    if (priceBufferRef.current.length > 100) {
      priceBufferRef.current = priceBufferRef.current.slice(-50);
    }

    // Clear any existing timer
    if (updateTimerRef.current) {
      clearTimeout(updateTimerRef.current);
    }

    // Only update if enough time has passed
    if (now - lastUpdateRef.current >= updateInterval) {
      // Use the latest price from buffer
      const latestPrice = priceBufferRef.current[priceBufferRef.current.length - 1];

      if (isMountedRef.current) {
        setCurrentPrice(latestPrice.price);
        onPriceUpdate?.(latestPrice);
      }

      // Clear buffer
      priceBufferRef.current = [];
      lastUpdateRef.current = now;
    } else {
      // Schedule an update for the remaining time
      const timeRemaining = updateInterval - (now - lastUpdateRef.current);
      updateTimerRef.current = setTimeout(() => {
        if (isMountedRef.current && priceBufferRef.current.length > 0) {
          const latestPrice = priceBufferRef.current[priceBufferRef.current.length - 1];
          setCurrentPrice(latestPrice.price);
          onPriceUpdate?.(latestPrice);
          priceBufferRef.current = [];
          lastUpdateRef.current = Date.now();
        }
      }, timeRemaining);
    }
  }, [updateInterval, onPriceUpdate]);

  // Connect to WebSocket
  useEffect(() => {
    isMountedRef.current = true;
    let pollInterval: NodeJS.Timeout | null = null;
    let isCleanedUp = false;

    const connectWebSocket = async () => {
      if (isCleanedUp) return;

      try {
        // Use existing backend WebSocket connection
        await backendMarketDataService.connectWebSocket(symbol, (data) => {
          if (!isCleanedUp && data.type === 'price' && data.symbol === symbol) {
            updatePrice({
              symbol: data.symbol,
              price: data.price,
              time: Date.now() / 1000,
              volume: data.volume,
              bid: data.bid,
              ask: data.ask,
            });
          }
        });

        if (!isCleanedUp) {
          setIsConnected(true);
        }
      } catch (error) {
        logger.error('WebSocket connection failed:', error);
        if (!isCleanedUp) {
          setIsConnected(false);
        }

        // Fallback to polling
        if (!isCleanedUp) {
          pollInterval = setInterval(async () => {
            if (isCleanedUp) return;

            try {
              const data = await backendMarketDataService.getCurrentMarketData(symbol);
              if (!isCleanedUp) {
                updatePrice({
                  symbol: data.symbol,
                  price: data.price,
                  time: Date.now() / 1000,
                });
              }
            } catch (err) {
              logger.error('Polling failed:', err);
            }
          }, 5000); // Poll every 5 seconds as fallback
        }
      }
    };

    connectWebSocket();

    // Cleanup function
    return () => {
      isCleanedUp = true;
      isMountedRef.current = false;

      // Clear any pending update timers
      if (updateTimerRef.current) {
        clearTimeout(updateTimerRef.current);
        updateTimerRef.current = null;
      }

      // Clear polling interval if it exists
      if (pollInterval) {
        clearInterval(pollInterval);
      }

      // Clear price buffer to free memory
      priceBufferRef.current = [];

      // Disconnect WebSocket
      backendMarketDataService.disconnectWebSocket();
      setIsConnected(false);
    };
  }, [symbol, updatePrice]);

  return {
    currentPrice,
    isConnected,
    priceBuffer: priceBufferRef.current,
  };
};
