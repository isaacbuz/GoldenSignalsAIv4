// This is an enhanced version of TickerAnalysisPanel with agent icons and theme-blended chart styling
import React from "react";
import { BarChart } from "lucide-react";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip, ResponsiveContainer } from "recharts";

const agentData = [
  { subject: "RSI", A: 80 },
  { subject: "MACD", A: 70 },
  { subject: "LSTM", A: 85 },
  { subject: "NLP", A: 65 },
  { subject: "IVRank", A: 60 },
];

export default function TickerAnalysisPanel() {
  return (
    // TickerAnalysisPanel shows a radar chart of agent strengths for the selected ticker.
    <div className="bg-bgPanel p-4 rounded-xl border border-borderSoft shadow font-sans" aria-label="Ticker analysis panel">
      <h3 className="text-lg font-semibold mb-2 flex items-center gap-2 text-white font-sans" aria-label="Strategy Summary">
        <BarChart size={18} className="text-accentGreen" /> Strategy Summary
      </h3>
      <ResponsiveContainer width="100%" height={250}>
        <RadarChart outerRadius={90} data={agentData} style={{ backgroundColor: "transparent" }}>
          <PolarGrid stroke="#444" strokeDasharray="3 3" />
          <PolarAngleAxis dataKey="subject" stroke="#bbb" tick={{ fill: "#ddd", fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#666" tick={{ fill: "#aaa", fontSize: 10 }} />
          <Radar
            name="AI Agent"
            dataKey="A"
            stroke="#00d97e"
            fill="#00d97e"
            fillOpacity={0.3}
            dot={{ r: 4, fill: "#00d97e" }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1c1f26", border: "1px solid #00d97e", color: "#fff" }}
            itemStyle={{ color: "#00d97e" }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}