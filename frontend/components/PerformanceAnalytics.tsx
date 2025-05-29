import React, { useEffect, useState } from 'react';

interface PerformanceData {
  winRate: number;
  pnl: number;
  totalTrades: number;
  backtest: { date: string; pnl: number }[];
}

export default function PerformanceAnalytics() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch('/api/performance')
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  return (
    <section className="p-4 bg-white dark:bg-zinc-900 rounded shadow mb-4">
      <h2 className="text-lg font-bold mb-2">Strategy Performance Analytics</h2>
      {loading && <div className="text-zinc-500">Loading...</div>}
      {data ? (
        <div>
          <div className="mb-2">Win Rate: <span className="font-bold text-green-600">{data.winRate}%</span></div>
          <div className="mb-2">Total Trades: <span className="font-bold">{data.totalTrades}</span></div>
          <div className="mb-2">Total P&L: <span className={`font-bold ${data.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>{data.pnl}</span></div>
          <div className="mb-2">Backtest Results:</div>
          <ul className="text-xs">
            {data.backtest.map((b, i) => (
              <li key={i} className="flex justify-between">
                <span>{b.date}</span>
                <span className={`ml-2 ${b.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>{b.pnl}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : !loading && <div className="text-zinc-600 dark:text-zinc-300">No analytics data available.</div>}
    </section>
  );
}
