import React from 'react';

function Admin() {
  return (
    <div className="admin-page">
      <h1>Admin Dashboard</h1>
      <p>Manage users, monitor system health, and control agents.</p>
      <section className="admin-agents">
        <h2>Agent Status</h2>
        <div className="agent-status-list">
          <div className="agent-status-card">Arbitrage Agent: <span className="agent-status agent-status-ok">Healthy</span></div>
          <div className="agent-status-card">Trading Agent: <span className="agent-status agent-status-warning">Warning</span></div>
        </div>
      </section>
      {/* User management and system controls will go here */}
    </div>
  );
}

export default Admin;
