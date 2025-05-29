import { useEffect, useState } from 'react';
import axios from 'axios';

const API_KEY = process.env.NEXT_PUBLIC_TWELVE_API_KEY;

interface ChartPoint {
  time: string;
  price: number;
  datetime?: string;
  close?: string;
}

interface UseTwelveDataResult {
  data: ChartPoint[];
  loading: boolean;
  error: string | null;
}

export function useTwelveData(symbol: string): UseTwelveDataResult {
  const [data, setData] = useState<ChartPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await axios.get(
          `https://api.twelvedata.com/time_series`,
          {
            params: {
              symbol,
              interval: '15min',
              outputsize: 30,
              apikey: API_KEY,
            },
          }
        );
        if (!data || !data.values || !Array.isArray(data.values)) {
          throw new Error('Malformed API response');
        }
        const formatted = data.values.reverse().map((d: any) => ({
          datetime: d.datetime,
          time: d.datetime.split(' ')[1],
          price: parseFloat(d.close),
          close: d.close,
        }));
        if (isMounted) {
          setData(formatted);
          setLoading(false);
        }
      } catch (err: any) {
        if (isMounted) {
          setError('API error: ' + (err?.message || 'Unknown error'));
          setData([]);
          setLoading(false);
        }
      }
    };
    fetchData();
    return () => { isMounted = false; };
  }, [symbol]);

  return { data, loading, error };
}

