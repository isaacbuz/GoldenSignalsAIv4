import React, { useState } from 'react';
import { useSignalWebSocket, AISignal } from '../hooks/useSignalWebSocket';

// Extend the AISignal type to support multi-agent outputs
type AgentOutput = {
  agent: string;
  trend?: string;
  action?: string;
  predicted_price?: number;
  confidence?: number;
  logits?: number[] | number[][];
  reason?: string;
  explanation?: string;
};

type MultiAgentAISignal = AISignal & {
  agents?: AgentOutput[];
  trend?: string;
  action?: string;
  agent?: string;
  predicted_price?: number;
  confidence?: number;
};

function isMultiAgent(signal: MultiAgentAISignal): signal is MultiAgentAISignal & { agents: AgentOutput[] } {
  return Array.isArray(signal.agents);
}

import SignalExplainModal from './SignalExplainModal';

const WS_URL = process.env.NEXT_PUBLIC_SIGNAL_WS_URL || 'ws://localhost:8000/ws/signals';

function ConfidenceBar({ confidence }: { confidence: number }) {
  return (
    <div className="w-full bg-zinc-200 dark:bg-zinc-800 rounded h-2 mt-1">
      <div
        className="bg-blue-500 h-2 rounded"
        style={{ width: `${Math.min(Math.max(confidence, 0), 1) * 100}%` }}
      />
    </div>
  );
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend?.toLowerCase() === 'bullish' || trend?.toLowerCase() === 'buy')
    return <span title="Bullish" className="text-green-500">▲</span>;
  if (trend?.toLowerCase() === 'bearish' || trend?.toLowerCase() === 'sell')
    return <span title="Bearish" className="text-red-500">▼</span>;
  return <span title="Hold/Flat" className="text-yellow-500">●</span>;
}

function ModelBadge({ agent }: { agent: string }) {
  let color = 'bg-gray-300 text-gray-800';
  if (agent.toLowerCase().includes('lstm')) color = 'bg-purple-200 text-purple-800';
  else if (agent.toLowerCase().includes('gru')) color = 'bg-pink-200 text-pink-800';
  else if (agent.toLowerCase().includes('cnn')) color = 'bg-blue-200 text-blue-800';
  else if (agent.toLowerCase().includes('attention')) color = 'bg-orange-200 text-orange-800';
  else if (agent.toLowerCase().includes('evolution')) color = 'bg-green-200 text-green-800';
  return (
    <span className={`ml-2 px-2 py-0.5 rounded text-xs font-semibold ${color}`}>{agent}</span>
  );
}

export default function SignalFeed() {
  const { signals, connected } = useSignalWebSocket(WS_URL);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<AISignal | null>(null);

  const handleSignalClick = (signal: AISignal) => {
    setSelectedSignal(signal);
    setModalOpen(true);
  };

  return (
    <section className="p-4 bg-white dark:bg-zinc-900 rounded shadow mb-4">
      <h2 className="text-lg font-bold mb-2">Live AI Signal Feed</h2>
      <div className="mb-2 text-xs text-zinc-500 flex items-center gap-2">
        <span className={connected ? 'text-green-600' : 'text-red-600'}>
          ●
        </span>
        {connected ? 'Connected to live feed' : 'Disconnected'}
      </div>
      <ul>
        {signals.length === 0 ? (
          <li className="text-zinc-600 dark:text-zinc-300">No signals yet. Waiting for live updates...</li>
        ) : (
          signals.map((signal: MultiAgentAISignal, idx) => (
            <li
              key={signal.id || idx}
              className="border border-zinc-200 dark:border-zinc-800 mb-4 p-4 rounded-lg bg-zinc-50 dark:bg-zinc-900 shadow hover:shadow-lg transition cursor-pointer"
              onClick={() => handleSignalClick(signal)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-lg">{signal.symbol}</span>
                <span className="text-xs text-zinc-500">{signal.time}</span>
              </div>
              {/* Multi-agent output display */}
              {isMultiAgent(signal) ? (
                <div className="space-y-2">
                  {signal.agents.map((agent, i) => (
                    <details key={agent.agent || i} className="bg-white dark:bg-zinc-800 rounded p-2 border border-zinc-100 dark:border-zinc-700">
                      <summary className="flex items-center gap-2 cursor-pointer">
                        <TrendIcon trend={agent.trend || agent.action} />
                        <span className="font-bold">{agent.trend || agent.action}</span>
                        <ModelBadge agent={agent.agent} />
                        <span className="ml-2 text-xs text-zinc-500">{agent.predicted_price !== undefined ? `Pred: $${Number(agent.predicted_price).toFixed(2)}` : ''}</span>
                        <span className="ml-2 text-xs text-blue-600 font-bold">{agent.confidence !== undefined ? `${(Number(agent.confidence) * 100).toFixed(1)}%` : ''}</span>
                      </summary>
                      <div className="mt-2">
                        <ConfidenceBar confidence={Number(agent.confidence)} />
                        <div className="text-xs text-zinc-500 mt-1">
                          {agent.logits && <div>Raw logits: <span className="font-mono">{JSON.stringify(agent.logits)}</span></div>}
                          {agent.reason && <div>Reason: {agent.reason}</div>}
                          {agent.explanation && <div>Explanation: {agent.explanation}</div>}
                        </div>
                      </div>
                    </details>
                  ))}
                </div>
              ) : (
                // fallback: single-agent (legacy)
                <div className="flex items-center gap-2">
                  <TrendIcon trend={signal.trend || signal.action} />
                  <span className="font-bold">{signal.trend || signal.action}</span>
                  <ModelBadge agent={signal.agent || 'UnknownModel'} />
                  <span className="ml-2 text-xs text-zinc-500">{signal.predicted_price !== undefined ? `Pred: $${Number(signal.predicted_price).toFixed(2)}` : ''}</span>
                  <span className="ml-2 text-xs text-blue-600 font-bold">{signal.confidence !== undefined ? `${(Number(signal.confidence) * 100).toFixed(1)}%` : ''}</span>
                  <ConfidenceBar confidence={Number(signal.confidence)} />
                </div>
              )}
            </li>
          ))
        )}
      </ul>
      <SignalExplainModal open={modalOpen} onClose={() => setModalOpen(false)} signal={selectedSignal} />
    </section>
  );
}
