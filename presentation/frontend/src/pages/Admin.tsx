import React, { useEffect, useState } from "react";
import AdminKpiCards from "../components/admin/AdminKpiCards";
import AdminRecentSignalsTable from "../components/admin/AdminRecentSignalsTable";
import SectorHeatmap from "../components/admin/SectorHeatmap";
import SectorTickerDrilldown from "../components/admin/SectorTickerDrilldown";
import AgentPerformancePanel from "../components/admin/AgentPerformancePanel";
import AgentControls from "../components/admin/AgentControls";
import AgentHealthMonitor from "../components/admin/AgentHealthMonitor";
import { toggleAgent, restartAgent, purgeSignals } from "../services/adminControlService";
import { authFetch } from "../services/authFetch";
import ExportButtons from "../components/ui/ExportButtons";
import ReportScheduler from "../components/admin/ReportScheduler";
import {
  fetchAdminStats,
  fetchRecentSignals,
  fetchSectors,
  fetchAgentStats,
} from "../services/adminService";

const AdminPanel: React.FC = () => {
  const [auditLog, setAuditLog] = useState<string[]>([]);
  // Optionally, replace with actual user from auth context
  const user = (typeof window !== 'undefined' && localStorage.getItem('username')) || 'admin';

  const logAction = (desc: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setAuditLog((log) => [`[${timestamp}] (${user}) ${desc}`, ...log].slice(0, 50));
  };

  const handleScheduleReport = async ({ type, format, frequency }: any) => {
    await authFetch("/admin/schedule-report", {
      method: "POST",
      body: JSON.stringify({ type, format, frequency }),
    });
    alert(`Scheduled ${type} report (${format}, ${frequency})`);
    logAction(`Scheduled ${type} report (${format}, ${frequency})`);
  };
  const [stats, setStats] = useState({
    totalSignals: 0,
    avgConfidence: 0,
    buyPercentage: 0,
    topAgent: "-",
  });

  const [recentSignals, setRecentSignals] = useState<{
    symbol: string;
    type: "buy" | "sell";
    confidence: number;
    timestamp: number;
    agents: string[];
}[]>([]);

  const [sectorData, setSectorData] = useState<{
    sector: string;
    buys: number;
    sells: number;
  }[]>([]);

  const [selectedSector, setSelectedSector] = useState<string | null>(null);
  const [tickers, setTickers] = useState<{
    symbol: string;
    lastSignal: "buy" | "sell" | "hold";
    confidence: number;
    agents: string[];
  }[]>([]);

  const handleSelectSector = (sector: string) => {
    setSelectedSector(sector);
    setTickers([
      { symbol: "AAPL", lastSignal: "buy", confidence: 0.92, agents: ["RSI", "LSTM"] },
      { symbol: "MSFT", lastSignal: "sell", confidence: 0.78, agents: ["MACD", "Flow"] },
      // Replace with API call later
    ]);
  };

  const [agentStats, setAgentStats] = useState<{
    name: string;
    signals: number;
    winRate: number;
    avgConfidence: number;
  }[]>([]);

  const [agents, setAgents] = useState<{
    id: string;
    name: string;
    enabled: boolean;
  }[]>([
    { id: "lstm", name: "LSTM", enabled: true },
    { id: "macd", name: "MACD Agent", enabled: true },
    { id: "finbert", name: "FinBERT", enabled: false },
  ]);

  const [agentHealth, setAgentHealth] = useState<any[]>([]);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await authFetch("/admin/agent-health");
        const json = await res.json();
        setAgentHealth(json);
      } catch (e) {
        // Optionally log or display error
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleToggle = async (id: string) => {
    await toggleAgent(id);
    setAgents((prev) =>
      prev.map((a) => (a.id === id ? { ...a, enabled: !a.enabled } : a))
    );
    logAction(`Toggled agent: ${id}`);
  };

  const handleRestart = async (id: string) => {
    await restartAgent(id);
    alert(`${id} restarted.`);
    logAction(`Restarted agent: ${id}`);
  };

  const handlePurge = async () => {
    await purgeSignals();
    alert("Signal cache purged.");
    logAction("Purged all signals");
  };






  useEffect(() => {
    const load = async () => {
      try {
        const stats = await fetchAdminStats();
        const recent = await fetchRecentSignals();
        const sectors = await fetchSectors();
        const agents = await fetchAgentStats();

        setStats(stats);
        setRecentSignals(recent);
        setSectorData(sectors);
        setAgentStats(agents);
      } catch (err) {
        console.error("Failed to load admin dashboard data:", err);
      }
    };
    load();
  }, []);

  return (
    <div className="text-white p-6">
      <h1 className="text-2xl font-bold mb-6">Admin Dashboard</h1>
      <AdminKpiCards stats={stats} />
      <ReportScheduler onSchedule={handleScheduleReport} />
      <ExportButtons data={recentSignals} title="recent-signals" />
      <AdminRecentSignalsTable data={recentSignals} />
      <SectorHeatmap data={sectorData} onSelectSector={handleSelectSector} />
      <SectorTickerDrilldown
        open={!!selectedSector}
        onClose={() => setSelectedSector(null)}
        sector={selectedSector || ""}
        tickers={tickers}
      />
      <ExportButtons data={agentStats} title="agent-stats" />
      <AgentPerformancePanel data={agentStats} />
      <AgentHealthMonitor agents={agentHealth} />
      <AgentControls
        agents={agents}
        onToggle={handleToggle}
        onRestart={handleRestart}
        onPurge={handlePurge}
      />
      <div className="mt-8">
        <ExportButtons data={auditLog.map((l) => ({ log: l }))} title="admin-audit-log" />
        <h3 className="text-sm font-semibold mb-2 text-gray-400">🧾 Admin Audit Log</h3>
        <div className="bg-gray-900 text-xs p-3 rounded max-h-40 overflow-y-auto font-mono border border-gray-700">
          {auditLog.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AdminPanel;
