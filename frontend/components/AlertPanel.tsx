import React, { useEffect, useState } from 'react';

interface Alert {
  id: string;
  message: string;
  time: string;
}

export default function AlertPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch('/api/alerts')
      .then(res => res.json())
      .then(data => setAlerts(data.alerts || []))
      .finally(() => setLoading(false));
  }, []);

  const acknowledgeAlert = async (id: string) => {
    setLoading(true);
    await fetch('/api/alerts', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id })
    });
    setAlerts(prev => prev.filter(a => a.id !== id));
    setLoading(false);
  };

  return (
    <section className="p-4 bg-white dark:bg-zinc-900 rounded shadow mb-4">
      <h2 className="text-lg font-bold mb-2">Alerts & Notifications</h2>
      <ul>
        {alerts.length === 0 ? (
          <li className="text-zinc-600 dark:text-zinc-300">No alerts yet.</li>
        ) : (
          alerts.map(alert => (
            <li key={alert.id} className="flex items-center justify-between py-1 border-b border-zinc-200 dark:border-zinc-800">
              <div>
                <span className="font-semibold">{alert.message}</span>
                <span className="ml-2 text-xs text-zinc-500">{alert.time}</span>
              </div>
              <button
                onClick={() => acknowledgeAlert(alert.id)}
                className="text-xs text-blue-600 hover:underline"
                disabled={loading}
              >Acknowledge</button>
            </li>
          ))
        )}
      </ul>
    </section>
  );
}
