import React, { useEffect, useState } from "react";
import axios from "axios";

interface HistoryPoint {
  score: number;
  trend: string;
  platform_breakdown: Record<string, number>;
  sample_post?: Record<string, string>;
  updated_at: string;
}

export default function TrendingSentimentHistory({ symbol }: { symbol: string }) {
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    axios
      .get(`/api/sentiment/history`, { params: { symbol } })
      .then((res) => setHistory(res.data))
      .catch((err) => setError(err?.message || 'Error fetching history'))
      .finally(() => setLoading(false));
  }, [symbol]);

  if (loading) return <div className="text-xs">Loading history…</div>;
  if (error) return <div className="text-xs text-red-500">{error}</div>;
  if (!history.length) return <div className="text-xs text-gray-400">No sentiment history found.</div>;

  return (
    <div className="mt-2">
      <div className="font-semibold text-xs mb-1">Sentiment History (last {history.length}):</div>
      <ul className="text-xs list-disc ml-4">
        {history.slice(0, 10).map((point, i) => (
          <li key={i}>
            <span className={
              point.trend === "bullish"
                ? "text-green-600"
                : point.trend === "bearish"
                ? "text-red-600"
                : "text-gray-600"
            }>
              {point.trend}
            </span>
            {": "}
            <b>{(point.score * 100).toFixed(1)}%</b>
            <span className="text-gray-400 ml-1">({new Date(point.updated_at).toLocaleString()})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
