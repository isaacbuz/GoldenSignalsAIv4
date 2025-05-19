// AdminUserManagement (migrated from JS, refactored for TypeScript & Tailwind)
import React from 'react';

const AdminUserManagement: React.FC = () => {
  // Placeholder: replace with real user management logic
  return (
    <div className="bg-white rounded shadow p-4">
      <h2 className="text-xl font-semibold mb-2">User Management</h2>
      <ul className="list-disc ml-6">
        <li>alice@example.com <span className="text-green-600">Active</span></li>
        <li>bob@example.com <span className="text-red-600">Suspended</span></li>
      </ul>
      <button className="bg-blue-600 text-white px-4 py-2 rounded mt-2 hover:bg-blue-700">Add User</button>
    </div>
  );
};

export default AdminUserManagement;
