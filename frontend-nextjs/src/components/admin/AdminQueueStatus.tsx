// AdminQueueStatus (migrated from JS, refactored for TypeScript & Tailwind)
import React from 'react';

const AdminQueueStatus: React.FC = () => {
  // Placeholder: replace with real queue status logic
  return (
    <div className="bg-white rounded shadow p-4">
      <h2 className="text-xl font-semibold mb-2">Queue Status</h2>
      <ul className="list-disc ml-6">
        <li>Inference Queue: <span className="text-green-600">Empty</span></li>
        <li>Training Queue: <span className="text-yellow-600">Processing (2 jobs)</span></li>
      </ul>
    </div>
  );
};

export default AdminQueueStatus;
