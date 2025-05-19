// AdminOnboardingModal (migrated from JS, refactored for TypeScript & Tailwind)
import React from 'react';

const AdminOnboardingModal: React.FC = () => {
  // Placeholder: replace with real onboarding modal logic
  return (
    <div className="fixed bottom-4 right-4 bg-white rounded shadow-lg p-4 w-80 border border-blue-200">
      <h2 className="text-lg font-bold mb-2">Welcome to the Admin Panel</h2>
      <p className="mb-2">Use this dashboard to monitor agents, review analytics, and manage users. For help, hover over any section or click the help icon.</p>
      <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full">Get Started</button>
    </div>
  );
};

export default AdminOnboardingModal;
