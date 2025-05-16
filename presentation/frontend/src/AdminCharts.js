// AdminCharts.js
// Purpose: Visualizes system performance metrics (CPU usage, memory usage, uptime, and active requests) for GoldenSignalsAI administrators. Fetches historical performance data from the backend and displays it using Chart.js. Charts are updated every minute for near real-time monitoring and capacity planning.

import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// AdminCharts component
function AdminCharts() {
  // State to store historical performance data
  const [history, setHistory] = useState([]);

  // Fetch historical performance data and update charts every minute
  useEffect(() => {
    // Fetch initial data
    fetch("/api/admin/performance/history")
      .then((res) => res.json())
      .then(setHistory);

    // Set up polling to update charts every minute
    const interval = setInterval(() => {
      fetch("/api/admin/performance/history")
        .then((res) => res.json())
        .then(setHistory);
    }, 60000); // update every minute

    // Clean up interval on unmount
    return () => clearInterval(interval);
  }, []);

  // If no data, display loading message
  if (!history.length) return <p>Loading charts...</p>;

  // Extract labels from historical data
  const labels = history.map((h) => new Date(h.timestamp * 1000).toLocaleTimeString());

  // CPU Usage Chart data
  const cpuData = {
    labels,
    datasets: [
      {
        label: "CPU Usage (%)",
        data: history.map((h) => h.cpu),
        borderColor: "#6be6c1",
        backgroundColor: "rgba(107,230,193,0.2)",
        tension: 0.3,
      },
    ],
  };

  // Memory Usage Chart data
  const memData = {
    labels,
    datasets: [
      {
        label: "Memory Usage (MB)",
        data: history.map((h) => h.memory),
        borderColor: "#f8b400",
        backgroundColor: "rgba(248,180,0,0.2)",
        tension: 0.3,
      },
    ],
  };

  // Uptime Chart data
  const uptimeData = {
    labels,
    datasets: [
      {
        label: "Uptime (s)",
        data: history.map((h) => h.uptime),
        borderColor: "#4fc3a1",
        backgroundColor: "rgba(79,195,161,0.2)",
        tension: 0.3,
      },
    ],
  };

  // Render three charts for CPU, memory, and uptime
  return (
    <div className="admin-charts">
      <h4>Performance Trends</h4>
      <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
        <div style={{ width: 320 }}>
          <Line data={cpuData} options={{ plugins: { legend: { display: false } } }} />
        </div>
        <div style={{ width: 320 }}>
          <Line data={memData} options={{ plugins: { legend: { display: false } } }} />
        </div>
        <div style={{ width: 320 }}>
          <Line data={uptimeData} options={{ plugins: { legend: { display: false } } }} />
        </div>
      </div>
    </div>
  );
}

export default AdminCharts;
