// AdminQueueStatus.js
// Purpose: Displays the status of the background task queue for GoldenSignalsAI administrators. Fetches and presents queue depth and active worker count by polling the backend every 5 seconds. Designed for real-time monitoring of system throughput and worker health.

import API_URL from './config';
import React, { useEffect, useState } from 'react';
import "./AdminPanel.css";

function AdminQueueStatus() {
  // State to store queue status from backend
  const [queue, setQueue] = useState(null);

  // Poll queue status every 5 seconds for real-time updates
  useEffect(() => {
    fetch(`${API_URL}/api/admin/queue`)
      .then((res) => res.json())
      .then(setQueue);
    const interval = setInterval(() => {
      fetch(`${API_URL}/api/admin/queue`)
        .then((res) => res.json())
        .then(setQueue);
    }, 5000); // update every 5 seconds

    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []);

  // If no queue data yet, show loading state
  if (!queue) return <p>Loading queue status...</p>;

  // Render queue depth and active worker count
  return (
    <div className="queue-status">
      <h4>Task Queue Status</h4>
      <div>Queue Depth: <b>{queue.depth}</b></div>
      <div>Active Workers: <b>{queue.active_workers}</b></div>
      <table>
        <tbody>
          <tr>
            <td>Active Workers</td>
            <td>{queue.active}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default AdminQueueStatus;
