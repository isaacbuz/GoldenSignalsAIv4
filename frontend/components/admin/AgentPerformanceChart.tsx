import React, { useEffect, useState } from "react";
// Using Recharts for simple charting
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from 'recharts';

interface FeedbackEntry {
  id: string;
  symbol: string;
  agent: string;
  action: string;
  rating: number;
  comment: string;
  timestamp: string;
}

// Aggregate agent performance by average rating over time (per day)
function aggregatePerformance(feedback: FeedbackEntry[]) {
  const byAgent: Record<string, Record<string, { sum: number; count: number }>> = {};
  feedback.forEach(fb => {
    const day = fb.timestamp.slice(0, 10);
    if (!byAgent[fb.agent]) byAgent[fb.agent] = {};
    if (!byAgent[fb.agent][day]) byAgent[fb.agent][day] = { sum: 0, count: 0 };
    byAgent[fb.agent][day].sum += fb.rating;
    byAgent[fb.agent][day].count += 1;
  });
  // For each agent, produce [{date, avg_rating}]
  const agentSeries: Record<string, { date: string; avg: number }[]> = {};
  Object.entries(byAgent).forEach(([agent, days]) => {
    agentSeries[agent] = Object.entries(days).map(([date, { sum, count }]) => ({ date, avg: sum / count }));
  });
  return agentSeries;
}

export default function AgentPerformanceChart() {
  const [feedback, setFeedback] = useState<FeedbackEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    fetch("/api/agents/feedback")
      .then(res => res.json())
      .then(data => {
        setFeedback(data.feedback || []);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to fetch feedback");
        setLoading(false);
      });
  }, []);

  const agentSeries = aggregatePerformance(feedback);
  // Build chart data: [{date, AgentA: avg, AgentB: avg, ...}]
  const allDates = Array.from(new Set(
    Object.values(agentSeries).flat().map(d => d.date)
  )).sort();
  const chartData = allDates.map(date => {
    const entry: any = { date };
    Object.entries(agentSeries).forEach(([agent, series]) => {
      const found = series.find(d => d.date === date);
      entry[agent] = found ? found.avg : null;
    });
    return entry;
  });

  return (
    <div style={{ marginTop: 32 }}>
      <h2>Agent Performance (Avg. Feedback Rating by Day)</h2>
      {loading ? <div>Loading...</div> : error ? <div className="error">{error}</div> : (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 20, right: 40, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis domain={[0, 5]} ticks={[1,2,3,4,5]} />
            <Tooltip />
            <Legend />
            {Object.keys(agentSeries).map(agent => (
              <Line key={agent} type="monotone" dataKey={agent} stroke={"#" + ((Math.abs(agent.split('').reduce((a, c) => a + c.charCodeAt(0), 0)) * 999999) % 0xffffff).toString(16).padStart(6, '0')} strokeWidth={2} connectNulls dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
