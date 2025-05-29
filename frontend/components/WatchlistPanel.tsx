import React, { useEffect, useState } from 'react';

export default function WatchlistPanel() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch('/api/watchlist')
      .then(res => res.json())
      .then(data => setSymbols(data.symbols || []))
      .finally(() => setLoading(false));
  }, []);

  const addSymbol = async () => {
    if (!input.trim()) return;
    setLoading(true);
    await fetch('/api/watchlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: input.trim().toUpperCase() })
    });
    setSymbols(prev => [...prev, input.trim().toUpperCase()]);
    setInput('');
    setLoading(false);
  };

  const removeSymbol = async (symbol: string) => {
    setLoading(true);
    await fetch('/api/watchlist', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    });
    setSymbols(prev => prev.filter(s => s !== symbol));
    setLoading(false);
  };

  return (
    <aside className="p-4 bg-white dark:bg-zinc-900 rounded shadow mb-4">
      <h2 className="text-lg font-bold mb-2">Watchlist</h2>
      <div className="flex gap-2 mb-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          className="rounded p-1 flex-1"
          placeholder="Add symbol (e.g. AAPL)"
        />
        <button
          onClick={addSymbol}
          className="bg-blue-600 text-white px-3 py-1 rounded disabled:opacity-50"
          disabled={loading || !input.trim()}
        >Add</button>
      </div>
      <ul>
        {symbols.length === 0 ? (
          <li className="text-zinc-600 dark:text-zinc-300">No symbols added yet.</li>
        ) : (
          symbols.map(symbol => (
            <li key={symbol} className="flex items-center justify-between py-1 border-b border-zinc-200 dark:border-zinc-800">
              <span>{symbol}</span>
              <button
                onClick={() => removeSymbol(symbol)}
                className="text-xs text-red-600 hover:underline"
                disabled={loading}
              >Remove</button>
            </li>
          ))
        )}
      </ul>
    </aside>
  );
}
