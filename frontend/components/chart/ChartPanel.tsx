/**
 * ChartPanel Component
 *
 * Displays a responsive line chart for the selected ticker symbol using data from TwelveData.
 * - Accessibility: Uses semantic HTML, ARIA roles/labels, and clear focus styles.
 * - Styling: All layout and colors use Tailwind CSS design tokens.
 * - Responsive: Layout adapts to screen size for optimal viewing.
 * - Documentation: Clear docstring for maintainability.
 */

import { useTickerContext } from '../../context/TickerContext';
import { useTwelveData } from '../../hooks/useTwelveData';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts';
import React from 'react';
import { generateAISignal } from '@/indicators/SignalAIEngine';

export default function ChartPanel() {
  const { selectedTicker } = useTickerContext();
  const { data, loading, error } = useTwelveData(selectedTicker);

  if (loading) {
    return (
      <section
        aria-busy="true"
        aria-live="polite"
        className="flex items-center justify-center min-h-[300px] bg-bgPanel rounded-xl shadow-neon p-8 font-sans"
      >
        <span className="text-center text-muted-foreground text-lg">Loading chart...</span>
      </section>
    );
  }

  if (error) {
    return (
      <section
        aria-live="assertive"
        className="flex items-center justify-center min-h-[300px] bg-bgPanel rounded-xl shadow-neon p-8 font-sans"
      >
        <span className="text-center text-red-500 text-lg">Error loading chart data.</span>
      </section>
    );
  }

  if (!data || data.length === 0) {
    return (
      <section
        className="flex items-center justify-center min-h-[300px] bg-bgPanel rounded-xl shadow-neon p-8 font-sans"
        aria-live="polite"
      >
        <span className="text-center text-muted-foreground text-lg">No data available for {selectedTicker}</span>
      </section>
    );
  }

  // AI Signal integration
  const aiSignal = React.useMemo(() => generateAISignal(data), [data]);
  React.useEffect(() => {
    if (aiSignal) {
      // Debug output
      console.log('AI Signal:', aiSignal);
    }
  }, [aiSignal]);

  return (
    <section
      role="region"
      aria-label={`${selectedTicker} trend chart`}
      className="bg-bgPanel p-4 md:p-6 rounded-xl shadow-neon transition-all duration-300 font-sans border border-borderSoft"
    >
      <header className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl md:text-3xl font-bold text-accentBlue tracking-tight font-sans" tabIndex={0}>
          {selectedTicker} Trend Chart
        </h2>
        {/* Add future controls here (like export, settings, etc.) */}
      </header>
      <div className="w-full h-[260px] md:h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 16, right: 24, left: 0, bottom: 0 }}
          >
            <XAxis
              dataKey="datetime"
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={{ stroke: '#334155' }}
            />
            <YAxis
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={{ stroke: '#334155' }}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', color: '#fff' }}
              labelStyle={{ color: '#38bdf8' }}
              itemStyle={{ color: '#fff' }}
            />
            <Line
              type="monotone"
              dataKey="close"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={false}
              aria-label="Closing price line"
            />
            {/* AI overlays */}
            {aiSignal?.channel && (
              <>
                <ReferenceLine y={aiSignal.channel.support} stroke="#22c55e" strokeDasharray="3 3" label="Support" />
                <ReferenceLine y={aiSignal.channel.resistance} stroke="#ef4444" strokeDasharray="3 3" label="Resistance" />
                <ReferenceLine y={aiSignal.channel.mid} stroke="#facc15" strokeDasharray="2 2" label="Mid" />
              </>
            )}
            {aiSignal?.supportResistance?.map((level, idx) => (
              <ReferenceLine key={`sr-${idx}`} y={level} stroke="#818cf8" strokeDasharray="1 3" label={`S/R ${idx+1}`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
        {/* Show AI trend direction */}
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs text-gray-400">AI Trend:</span>
          <span className={`font-bold ${aiSignal?.trend === 'up' ? 'text-green-400' : aiSignal?.trend === 'down' ? 'text-red-400' : 'text-yellow-400'}`}>{aiSignal?.trend?.toUpperCase()}</span>
          <span className="text-xs text-gray-400 ml-4">Confidence:</span>
          <span className="font-bold text-accentBlue">{aiSignal?.confidence}</span>
        </div>
      </div>
    </section>
  );
}
