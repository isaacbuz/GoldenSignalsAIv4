import { useState, useEffect } from 'react';

export function usePriceFeed(ticker: string) {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    // Replace with real API
    const mock = [
      { time: '09:30', price: 173 },
      { time: '10:30', price: 175 },
    ];
    setTimeout(() => setData(mock), 500);
  }, [ticker]);

  return data;
}
