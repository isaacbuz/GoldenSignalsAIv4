import { useEffect, useState } from 'react';

export function useMockLiveChart(ticker: string) {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    const mock = Array.from({ length: 6 }, (_, i) => ({
      time: `${9 + i}:30`,
      price: 170 + Math.floor(Math.random() * 10),
    }));
    setTimeout(() => setData(mock), 500);
  }, [ticker]);

  return data;
}
