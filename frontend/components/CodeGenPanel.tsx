import { useState } from 'react';

/**
 * CodeGenPanel lets users describe a trading strategy in plain English and converts it to JSON rules using the codegen API.
 */

export default function CodeGenPanel() {
  const [input, setInput] = useState('');
  const [json, setJson] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generate = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/strategy/codegen', {
        method: 'POST',
        body: JSON.stringify({ text: input }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error('Failed to generate code');
      const data = await res.json();
      setJson(data.json);
    } catch (err) {
      setError('Error generating code. Please try again.');
      setJson('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-bgPanel text-white p-6 md:p-8 rounded-xl shadow font-sans">
      <h2 className="text-xl font-bold text-accentPink mb-4">Strategy to Code</h2>
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="w-full p-3 rounded-lg bg-bgDark mb-4 h-40 font-mono text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accentBlue"
        placeholder="Describe your strategy in plain English..."
        aria-label="Strategy description"
      />
      <button
        onClick={generate}
        className="w-full bg-accentGreen text-black py-2 font-bold rounded-lg shadow-neon hover:shadow-glow transition"
        aria-label="Convert to Code"
        disabled={loading}
      >
        {loading ? 'Converting...' : 'Convert to Code'}
      </button>
      {error && <div className="text-red-400 mt-3">{error}</div>}
      {json && (
        <pre className="mt-4 bg-black text-green-400 p-4 rounded text-sm overflow-x-auto" aria-label="Generated JSON output">
          {json}
        </pre>
      )}
    </div>
  );
}
