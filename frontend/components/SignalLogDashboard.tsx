import React, { useEffect, useState } from "react";
import type { SignalLog } from "./signalTypes";
import toast from 'react-hot-toast';

export default function SignalLogDashboard() {
  const [logs, setLogs] = useState<SignalLog[]>([]);
  const [minConfidence, setMinConfidence] = useState(0);
  const [strategyFilter, setStrategyFilter] = useState('');
  const [agentFilter, setAgentFilter] = useState('');

  async function fetchLogs() {
    const params = new URLSearchParams();
    if (minConfidence) params.append('min_confidence', minConfidence.toString());
    if (strategyFilter) params.append('strategy', strategyFilter);
    if (agentFilter) params.append('agent', agentFilter);
    try {
      const res = await fetch(`/api/logs?${params.toString()}`);
      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      if (!data || !Array.isArray(data.results)) throw new Error('Malformed data');
      setLogs(data.results);
    } catch (err: any) {
      setLogs([]);
      toast.error('Failed to load signal logs: ' + (err?.message || 'Unknown error'));
    }
  }

  useEffect(() => {
    fetchLogs();
  }, []);

  return (
    <div className="bg-bgPanel min-h-screen text-white p-6 font-sans" aria-label="Signal log dashboard">
      <h2 className="text-2xl font-bold mb-4 font-sans" aria-label="Signal Log Dashboard"> Signal Log Dashboard</h2>
      <div className="flex flex-wrap gap-4 mb-6">
        <input placeholder="Strategy..." className="bg-bgDark px-2 py-1 rounded border border-borderSoft text-white font-sans" value={strategyFilter} onChange={e => setStrategyFilter(e.target.value)} aria-label="Strategy filter" />
        <input placeholder="Agent..." className="bg-bgDark px-2 py-1 rounded border border-borderSoft text-white font-sans" value={agentFilter} onChange={e => setAgentFilter(e.target.value)} aria-label="Agent filter" />
        <input type="number" placeholder="Min Confidence" className="bg-bgDark px-2 py-1 rounded border border-borderSoft text-white font-sans w-[150px]" value={minConfidence} onChange={e => setMinConfidence(parseInt(e.target.value))} aria-label="Minimum confidence filter" />
        <button className="bg-accentGreen px-4 py-1 rounded font-bold hover:bg-accentGreen/80 transition" onClick={fetchLogs} aria-label="Apply filters">Filter</button>
      </div>
      <div className="bg-bgDark border border-borderSoft rounded p-4 space-y-4">
        {logs.length === 0 && <p className="text-red-400">No logs found or failed to load. Try adjusting filters or check your connection.</p>}
        {logs.map((log, i) => (
          <div key={i} className="border-b border-borderSoft pb-2">
            <p><strong>{log.timestamp}</strong> | {log.ticker} | <span className="text-accentGreen">{log.blended.strategy}</span> | Confidence: {log.blended.confidence}%</p>
            <p className="text-sm text-gray-400">{log.blended.explanation}</p>
            <details className="text-sm mt-1">
              <summary className="cursor-pointer text-accentBlue">Agents</summary>
              <ul className="ml-4 list-disc">
                {Object.entries(log.agents).map(([name, data]) => (
                  <li key={name}><strong>{name}:</strong> {data.explanation}</li>
                ))}
              </ul>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
}
