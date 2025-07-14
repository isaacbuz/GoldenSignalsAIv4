import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { tradingAPI } from '../services/api/tradingApi';
import type { PriceData, Signal, NewsItem, OptionsFlow } from '../services/api/tradingApi';

// Hook for live price data
export const useLivePriceData = (symbol: string, timeframe: string) => {
  const [realtimePrice, setRealtimePrice] = useState<PriceData | null>(null);

  const { data: priceData, isLoading, error } = useQuery({
    queryKey: ['priceData', symbol, timeframe],
    queryFn: () => tradingAPI.getPriceData(symbol, timeframe),
    enabled: !!symbol,
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  useEffect(() => {
    if (!symbol) return;

    // Connect to WebSocket for real-time updates
    const cleanup = tradingAPI.connectToRealTimeData(symbol, (update) => {
      if (update.type === 'price') {
        setRealtimePrice(update.data);
      }
    });

    return cleanup;
  }, [symbol]);

  return {
    priceData: priceData || [],
    realtimePrice,
    isLoading,
    error,
  };
};

// Hook for signal generation
export const useSignalGeneration = () => {
  const mutation = useMutation({
    mutationFn: ({ symbol, timeframe }: { symbol: string; timeframe: string }) =>
      tradingAPI.generateSignal(symbol, timeframe),
  });

  return {
    generateSignal: mutation.mutate,
    isGenerating: mutation.isPending,
    signal: mutation.data,
    error: mutation.error,
  };
};

// Hook for latest signals
export const useLiveSignals = (symbol?: string) => {
  const [realtimeSignal, setRealtimeSignal] = useState<Signal | null>(null);

  const { data: signals, isLoading, error } = useQuery({
    queryKey: ['signals', symbol],
    queryFn: () => tradingAPI.getLatestSignals(symbol),
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  useEffect(() => {
    if (!symbol) return;

    // Connect to WebSocket for real-time signal updates
    const cleanup = tradingAPI.connectToRealTimeData(symbol, (update) => {
      if (update.type === 'signal') {
        setRealtimeSignal(update.data);
      }
    });

    return cleanup;
  }, [symbol]);

  return {
    signals: signals || [],
    realtimeSignal,
    isLoading,
    error,
  };
};

// Hook for news feed
export const useLiveNews = (symbol: string) => {
  const { data: news, isLoading, error } = useQuery({
    queryKey: ['news', symbol],
    queryFn: () => tradingAPI.getNews(symbol),
    enabled: !!symbol,
    refetchInterval: 60000, // Refetch every minute
  });

  return {
    news: news || [],
    isLoading,
    error,
  };
};

// Hook for options flow
export const useLiveOptionsFlow = (symbol: string) => {
  const { data: optionsFlow, isLoading, error } = useQuery({
    queryKey: ['optionsFlow', symbol],
    queryFn: () => tradingAPI.getOptionsFlow(symbol),
    enabled: !!symbol,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  return {
    optionsFlow: optionsFlow || [],
    isLoading,
    error,
  };
};

// Hook for market metrics
export const useLiveMarketMetrics = () => {
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ['marketMetrics'],
    queryFn: () => tradingAPI.getMarketMetrics(),
    refetchInterval: 15000, // Refetch every 15 seconds
  });

  return {
    metrics: metrics || {},
    isLoading,
    error,
  };
};