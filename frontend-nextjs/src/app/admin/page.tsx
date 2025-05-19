// Admin Panel Page (Next.js, TypeScript, Tailwind CSS)
// Migrated and modernized from legacy AdminPanel.js

import React from 'react';
import AdminAgentControls from '../../components/admin/AdminAgentControls';
import AdminAgentHealth from '../../components/admin/AdminAgentHealth';
import AdminAlerts from '../../components/admin/AdminAlerts';
import AdminCharts from '../../components/admin/AdminCharts';
import AdminOnboardingModal from '../../components/admin/AdminOnboardingModal';
import AdminQueueStatus from '../../components/admin/AdminQueueStatus';
import AdminUserManagement from '../../components/admin/AdminUserManagement';

export default function AdminPanelPage() {
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6 text-center text-blue-900">Admin Panel</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-6">
          <AdminAgentControls />
          <AdminAgentHealth />
          <AdminAlerts />
        </div>
        <div className="space-y-6">
          <AdminCharts />
          <AdminQueueStatus />
          <AdminUserManagement />
        </div>
      </div>
      <AdminOnboardingModal />
    </div>
  );
}
