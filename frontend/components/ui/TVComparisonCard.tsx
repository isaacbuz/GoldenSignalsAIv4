import React from "react";

type SignalSet = {
  name: string;
  signal: string;
  confidence?: number;
};

type Props = {
  symbol: string;
  agentSignals: SignalSet[];
  tradingViewSignal: SignalSet;
};

const TVComparisonCard = ({ symbol, agentSignals, tradingViewSignal }: Props) => {
  const agreementCount = agentSignals.filter(
    (s) => s.signal === tradingViewSignal.signal
  ).length;

  return (
    <div className="bg-[#1f2937] text-white p-4 rounded shadow-md space-y-2">
      <h3 className="text-lg font-semibold text-accentBlue">📊 AI Signal Comparison</h3>
      <p className="text-sm text-gray-400">Symbol: {symbol}</p>
      <div className="text-sm">
        <p>🔵 TradingView Signal: <span className="text-accentPink font-medium">{tradingViewSignal.signal}</span></p>
        <p>🤖 Agent Signals:</p>
        <ul className="ml-4 list-disc">
          {agentSignals.map((s, i) => (
            <li key={i}>{s.name}: {s.signal} ({s.confidence || "?"}%)</li>
          ))}
        </ul>
        <p className="mt-2 text-green-400">Agreement: {agreementCount} / {agentSignals.length} agents</p>
      </div>
    </div>
  );
};

export default TVComparisonCard;
