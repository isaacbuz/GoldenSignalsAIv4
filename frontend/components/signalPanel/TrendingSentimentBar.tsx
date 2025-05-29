import React, { useEffect, useState, Suspense, lazy } from "react";
import axios from "axios";
import { Tooltip } from "react-tooltip";

interface SentimentData {
  symbol: string;
  score: number;
  trend: string;
  platform_breakdown: Record<string, number>;
  sample_post?: Record<string, string>; // platform -> sample post
  youtube_links?: string[]; // optional: array of YouTube URLs
}

const PLATFORM_COLORS: Record<string, string> = {
  X: '#1DA1F2',
  LinkedIn: '#0077b5',
  Reddit: '#FF4500',
  Facebook: '#1877F3',
  YouTube: '#FF0000',
  StockTwits: '#5C5457',
  TikTok: '#010101',
  HackerNews: '#FF6600',
};

function getPlatformColor(platform: string) {
  return PLATFORM_COLORS[platform] || '#888';
}

function formatPercent(score: number) {
  return `${(score * 100).toFixed(1)}%`;
}

import { useTickerContext } from '../../context/TickerContext';

const HistoricalSentiment = lazy(() => import('./TrendingSentimentHistory'));

export default function TrendingSentimentBar({ symbols: propSymbols = ["AAPL", "TSLA", "MSFT"] }) {
  const { tickers } = useTickerContext ? useTickerContext() : { tickers: propSymbols };
  const symbols = tickers || propSymbols;
  const [recommendations, setRecommendations] = useState<SentimentData[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    axios
      .get("/api/sentiment/recommend", {
        params: { symbols: symbols, direction: "bullish" },
      })
      .then((res) => setRecommendations(res.data))
      .catch((err) => setError(err?.message || 'Error fetching sentiment'))
      .finally(() => setLoading(false));
  }, [symbols]);

  const handleToggle = (symbol: string) => {
    setExpanded((prev) => ({ ...prev, [symbol]: !prev[symbol] }));
  };

  if (loading) return <div className="p-4">Loading trending sentiment…</div>;
  if (error) return <div className="p-4 text-red-500">{error}</div>;

  return (
    <div className="p-4 rounded-xl shadow bg-white dark:bg-zinc-800 w-full" aria-label="Trending Sentiment Bar">
      <h2 className="text-xl font-bold mb-2">🔥 Trending Bullish Stocks (Sentiment)</h2>
      <div className="mb-2 flex gap-4 text-xs">
        <span className="font-bold">Legend:</span>
        {Object.keys(PLATFORM_COLORS).map((platform) => (
          <span key={platform} className="flex items-center gap-1">
            <span style={{ background: getPlatformColor(platform), width: 12, height: 12, display: 'inline-block', borderRadius: 2 }} />
            {platform}
          </span>
        ))}
      </div>
      <ul className="space-y-4">
        {recommendations.map((rec) => (
          <li key={rec.symbol} className="flex flex-col gap-1 border-b pb-2">
            <div className="flex items-center justify-between">
              <button
                className="text-lg font-medium hover:underline focus:outline-none"
                onClick={() => handleToggle(rec.symbol)}
                aria-expanded={!!expanded[rec.symbol]}
                aria-controls={`panel-${rec.symbol}`}
                tabIndex={0}
                onKeyDown={e => {
                  if (e.key === 'Enter' || e.key === ' ') handleToggle(rec.symbol);
                }}
              >
                {rec.symbol}
              </button>
              <div className="flex items-center gap-2">
                <div className={`text-sm ${rec.trend === "bullish" ? "text-green-500" : "text-red-500"}`}>{rec.trend}</div>
                <div className="text-xs text-gray-400">{formatPercent(rec.score)}</div>
              </div>
            </div>
            <div className="w-full flex items-center gap-1 mt-1">
              {Object.entries(rec.platform_breakdown).map(([platform, score]) => (
  <div
    key={platform}
    data-tooltip-id={`sentiment-bar-tooltip`}
    data-tooltip-content={`Platform: ${platform}\nSentiment: ${formatPercent(score)}${rec.sample_post && rec.sample_post[platform] ? `\nSample: ${rec.sample_post[platform]}` : ''}`}
    style={{
      width: `${Math.max(score * 100, 5)}%`,
      background: getPlatformColor(platform),
      height: 10,
      borderRadius: 4,
      marginRight: 2,
      cursor: platform === 'YouTube' && rec.youtube_links && rec.youtube_links.length > 0 ? 'pointer' : 'default',
    }}
    onClick={() => {
      if (platform === 'YouTube' && rec.youtube_links && rec.youtube_links.length > 0) {
        window.open(rec.youtube_links[0], '_blank');
      }
    }}
    aria-label={`Sentiment bar for ${platform}`}
    tabIndex={0}
    onKeyDown={e => {
      if ((e.key === 'Enter' || e.key === ' ') && platform === 'YouTube' && rec.youtube_links && rec.youtube_links.length > 0) {
        window.open(rec.youtube_links[0], '_blank');
      }
    }}
  />
))}
<Tooltip id="sentiment-bar-tooltip" />
              
            </div>
            {expanded[rec.symbol] && (
              <div id={`panel-${rec.symbol}`} className="mt-2 bg-gray-100 dark:bg-zinc-700 rounded p-2" aria-live="polite">
                <div className="font-semibold mb-1">Per-Platform Breakdown:</div>
                <ul className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(rec.platform_breakdown).map(([platform, score]) => (
                    <li key={platform}>
                      <span style={{ color: getPlatformColor(platform), fontWeight: 700 }}>{platform}:</span>{' '}
                      {formatPercent(score)}
                      {platform === 'YouTube' && rec.youtube_links && rec.youtube_links.length > 0 && (
                        <>
                          {' '}
                          <a
                            href={rec.youtube_links[0]}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-500 underline ml-1"
                          >
                            Watch
                          </a>
                        </>
                      )}
                    </li>
                  ))}
                </ul>
                {rec.sample_post && (
                  <div className="mt-2">
                    <div className="font-semibold">Sample Posts:</div>
                    <ul className="list-disc ml-4">
                      {Object.entries(rec.sample_post).map(([platform, post]) => (
                        <li key={platform}>
                          <span style={{ color: getPlatformColor(platform), fontWeight: 700 }}>{platform}:</span>{' '}
                          <span className="italic">{post}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <Suspense fallback={<div>Loading historical sentiment…</div>}>
                  <HistoricalSentiment symbol={rec.symbol} />
                </Suspense>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
