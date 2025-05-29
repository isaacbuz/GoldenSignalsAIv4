import React, { useState, Suspense } from "react";
import ProDashboardLayout from "../components/layout/ProDashboardLayout";
import SignalFeed from "../components/SignalFeed";
import WatchlistPanel from "../components/WatchlistPanel";
import AlertPanel from "../components/AlertPanel";
import PerformanceAnalytics from "../components/PerformanceAnalytics";
import SignalExplainModal from "../components/SignalExplainModal";
import SignalChartPanel from "../components/SignalChartPanel";
import SignalLogDashboard from "../components/SignalLogDashboard";
import StrategyBuilder from "../components/StrategyBuilder";
import ConfidenceHeatmap from "../components/ui/ConfidenceHeatmap";
import UserPreferences from "../components/UserPreferences";
import AnalyticsSection from "../components/AnalyticsSection";
import { AgentTemplate } from "../components/agent/AgentTemplate";
const TVComparisonCard = React.lazy(() => import('../components/ui/TVComparisonCard'));

import { useAgentRegistry } from "../components/agent/AgentOrchestrator";

function AdminPanel() {
  const { agents } = useAgentRegistry();
  return (
    <div className="p-8 text-xl font-bold text-neon-green bg-zinc-900 rounded-xl shadow">
      <h2 className="text-2xl mb-4">Admin Dashboard</h2>
      <div className="space-y-2">
        <div className="font-semibold">Registered UI Agents:</div>
        <ul className="list-disc pl-6">
          {Object.keys(agents).length === 0 && <li className="text-zinc-400">No agents registered</li>}
          {Object.keys(agents).map(name => (
            <li key={name} className="text-neon-green">{name}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function Home() {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  // Wire up modal and symbol selection
  const handleSignalClick = (signal: any) => {
    setSelectedSignal(signal);
    setModalOpen(true);
  }
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<any>(null);

  return (
    <ProDashboardLayout activeSection={activeSection} onSectionChange={setActiveSection}>
      {activeSection === 'dashboard' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Watchlist & Alerts */}
          <div className="space-y-4">
            <AgentTemplate name="WatchlistAgent">
              <WatchlistPanel />
            </AgentTemplate>
            <AgentTemplate name="AlertAgent">
              <AlertPanel />
            </AgentTemplate>
          </div>
          {/* Center: Signal Feed & Chart */}
          <div className="col-span-2 space-y-4">
            <AgentTemplate name="SignalFeedAgent">
              <SignalFeed />
            </AgentTemplate>
            <AgentTemplate name="PerformanceAgent">
              <PerformanceAnalytics />
            </AgentTemplate>
            <AgentTemplate name="SignalChartAgent">
              <SignalChartPanel symbol={selectedSymbol} signals={[]} />
            </AgentTemplate>
          </div>
        </div>
      )}
      {activeSection === 'watchlist' && (
        <AgentTemplate name="WatchlistAgent">
          <WatchlistPanel />
        </AgentTemplate>
      )}
      {activeSection === 'signal-log' && (
        <AgentTemplate name="SignalLogAgent">
          <SignalLogDashboard />
        </AgentTemplate>
      )}
      {activeSection === 'strategy' && (
        <AgentTemplate name="StrategyAgent">
          <StrategyBuilder onUpdate={() => {}} />
        </AgentTemplate>
      )}
      {activeSection === 'analytics' && (
        <>
          <AgentTemplate name="AnalyticsAgent">
            <AnalyticsSection />
          </AgentTemplate>
        </>
      )}
      {activeSection === 'admin' && <AdminPanel />}
      {activeSection === 'settings' && <UserPreferences />}
      <SignalExplainModal open={modalOpen} onClose={() => setModalOpen(false)} signal={selectedSignal} />
    </ProDashboardLayout>
  );
}
