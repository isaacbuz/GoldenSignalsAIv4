import React, { useEffect, useState } from "react";
import "./AdminPanel.css";

function AdminQueueStatus() {
  const [queue, setQueue] = useState(null);

  useEffect(() => {
    fetch("/api/admin/queue")
      .then((res) => res.json())
      .then(setQueue);
    const interval = setInterval(() => {
      fetch("/api/admin/queue")
        .then((res) => res.json())
        .then(setQueue);
    }, 5000); // update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (!queue) return <p>Loading queue status...</p>;

  return (
    <div className="queue-status">
      <h4>Queue & Task Monitoring</h4>
      <table className="queue-status-table">
        <tbody>
          <tr>
            <td>Queue Depth</td>
            <td>{queue.depth}</td>
          </tr>
          <tr>
            <td>Workers</td>
            <td>{queue.workers}</td>
          </tr>
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
