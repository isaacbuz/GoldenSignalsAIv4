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

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function AdminCharts() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch("/api/admin/performance/history")
      .then((res) => res.json())
      .then(setHistory);
    const interval = setInterval(() => {
      fetch("/api/admin/performance/history")
        .then((res) => res.json())
        .then(setHistory);
    }, 60000); // update every minute
    return () => clearInterval(interval);
  }, []);

  if (!history.length) return <p>Loading charts...</p>;

  const labels = history.map((h) => new Date(h.timestamp * 1000).toLocaleTimeString());

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
