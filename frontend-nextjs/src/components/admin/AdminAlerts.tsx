// AdminAlerts (migrated from JS, refactored for TypeScript & Tailwind)
import React from 'react';

const AdminAlerts: React.FC = () => {
  // Placeholder: replace with real alert data
  return (
    <div className="bg-white rounded shadow p-4">
      <h2 className="text-xl font-semibold mb-2">Alerts</h2>
      <ul className="list-disc ml-6">
        <li className="text-red-600">Critical: Model drift detected in LSTM</li>
        <li className="text-yellow-600">Warning: API quota near limit</li>
        <li className="text-green-600">Info: All agents synced</li>
      </ul>
    </div>
  );
};

export default AdminAlerts;
