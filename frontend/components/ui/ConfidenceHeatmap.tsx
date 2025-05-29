import React from 'react';

interface HeatmapProps {
  data: Array<{ symbol: string; confidence: number; timestamp: string }>;
}

// Simple color scale: low = red, mid = yellow, high = green
function getColor(confidence: number) {
  if (confidence >= 80) return '#22c55e'; // green
  if (confidence >= 60) return '#eab308'; // yellow
  return '#ef4444'; // red
}

export default function ConfidenceHeatmap({ data }: HeatmapProps) {
  return (
    <div className="w-full overflow-x-auto">
      <table className="min-w-full text-center border border-gray-700 rounded">
        <thead>
          <tr className="bg-gray-800">
            <th className="px-3 py-2">Symbol</th>
            <th className="px-3 py-2">Confidence</th>
            <th className="px-3 py-2">Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <td className="px-3 py-2">{row.symbol}</td>
              <td className="px-3 py-2">
                <span style={{
                  background: getColor(row.confidence),
                  color: '#fff',
                  padding: '4px 12px',
                  borderRadius: 6,
                  fontWeight: 600,
                }}>{row.confidence}%</span>
              </td>
              <td className="px-3 py-2 text-xs text-gray-400">{row.timestamp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
