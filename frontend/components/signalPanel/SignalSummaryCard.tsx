// Placeholder for SignalSummaryCard.
import { useTickerContext } from '../../context/TickerContext';
import { useTwelveData } from '../../hooks/useTwelveData';
import { useSignalAI } from '../../hooks/useSignalAI';
import { getAdaptiveOscillatorScore } from '@/indicators/AdaptiveOscillator';
import { AIScoreDisplay } from './AIScoreDisplay';

export default function SignalSummaryCard() {
  const { selected } = useTickerContext();
  const { data, loading, error } = useTwelveData(selected);
  const signal = useSignalAI(data);

  return (
    <div className="bg-bgPanel p-5 rounded-xl shadow-glow animate-fadeIn font-sans">
      <h3 className="text-lg font-semibold text-accentPink mb-2"> AI Signal</h3>
      {loading ? (
        <div className="text-gray-400 animate-pulse">Loading signal...</div>
      ) : error ? (
        <div className="text-red-400">Error loading signal: {error}</div>
      ) : signal ? (
        <>
          <div className="text-xl font-bold text-white">{signal.type}</div>
          <div className="text-sm text-gray-400 mb-2">{signal.explanation}</div>
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
            <div>Entry: <span className="text-white">${signal.entry}</span></div>
            <div>Confidence: <span className="text-accentGreen">{signal.confidence}%</span></div>
          </div>
          {/* AI Oscillator Score */}
          <AIScoreDisplay data={data ?? []} />
        </>
      ) : (
        <p className="text-gray-500">No active signal</p>
      )}
      <div className="text-xs text-gray-500 mt-3 italic">Live feed powered by OpenAI + TwelveData</div>
    </div>
  );
}
