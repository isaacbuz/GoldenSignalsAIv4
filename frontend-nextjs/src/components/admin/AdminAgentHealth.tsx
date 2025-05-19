// AdminAgentHealth (migrated from JS, refactored for TypeScript & Tailwind)
import React from 'react';

const AdminAgentHealth: React.FC = () => {
  // Placeholder: replace with real agent health data
  return (
    <div className="bg-white rounded shadow p-4">
      <h2 className="text-xl font-semibold mb-2">Agent Health</h2>
      <ul className="list-disc ml-6">
        <li>Quant Agent: <span className="text-green-600">Healthy</span></li>
        <li>Macro Agent: <span className="text-yellow-600">Warning</span></li>
        <li>Momentum Agent: <span className="text-red-600">Offline</span></li>
      </ul>
    </div>
  );
};

export default AdminAgentHealth;
