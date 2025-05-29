/* @jsxRuntime automatic */
import React, { ReactNode, useState } from 'react';
import {
  FaChartLine,
  FaListAlt,
  FaCogs,
  FaUserShield,
  FaBell,
  FaCog,
  FaMoon,
  FaSun
} from 'react-icons/fa';

interface ProDashboardLayoutProps {
  children: ReactNode;
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const sections = [
  { key: 'dashboard', label: 'Dashboard', icon: FaChartLine },
  { key: 'watchlist', label: 'Watchlist', icon: FaListAlt },
  { key: 'signal-log', label: 'Signal Log', icon: FaListAlt },
  { key: 'strategy', label: 'Strategy Builder', icon: FaCogs },
  { key: 'analytics', label: 'Analytics', icon: FaChartLine },
  { key: 'admin', label: 'Admin', icon: FaUserShield },
  { key: 'settings', label: 'Settings', icon: FaCog },
];

export default function ProDashboardLayout({ children, activeSection, onSectionChange }: ProDashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className={`flex h-screen w-screen ${darkMode ? 'dark bg-zinc-900 text-white' : 'bg-zinc-50 text-zinc-900'}`}>
      {/* Sidebar */}
      <aside className={`flex flex-col h-full bg-white dark:bg-zinc-900 shadow-lg transition-all duration-200 ${sidebarOpen ? 'w-56' : 'w-16'} z-20`}>
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-800">
          <span className="font-extrabold text-xl tracking-tight text-neon-green">GS-AI</span>
          <button onClick={() => setSidebarOpen(s => !s)} className="ml-2 p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800">
            {sidebarOpen ? <span>&lt;</span> : <span>&gt;</span>}
          </button>
        </div>
        <nav className="flex-1 flex flex-col mt-4 gap-2">
          {sections.map(s => {
            const Icon = s.icon as React.ElementType;
            return (
              <button
                key={s.key}
                className={`flex items-center gap-3 px-4 py-2 rounded transition font-medium ${activeSection === s.key ? 'bg-neon-green/20 text-neon-green' : 'hover:bg-zinc-100 dark:hover:bg-zinc-800'}`}
                onClick={() => onSectionChange(s.key)}
                aria-current={activeSection === s.key ? 'page' : undefined}
              >
                <span className="text-lg"><Icon /></span>
                {sidebarOpen && <span>{s.label}</span>}
              </button>
            );
          })}
        </nav>
        <div className="p-4 border-t border-zinc-200 dark:border-zinc-800 flex gap-2 items-center justify-between">
          <button onClick={() => setDarkMode(d => !d)} className="p-2 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800">
            {darkMode ? <FaSun /> : <FaMoon />}
          </button>
          {sidebarOpen && <span className="text-xs text-zinc-500">Theme</span>}
        </div>
      </aside>
      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-y-auto">
        {/* Topbar */}
        <header className="flex items-center justify-between px-6 py-3 bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800 shadow-sm">
          <div className="flex items-center gap-3">
            <span className="font-bold text-xl text-neon-green">GoldenSignalsAI</span>
            <input
              type="text"
              placeholder="Search symbols..."
              className="ml-4 px-2 py-1 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-white focus:outline-none"
            />
          </div>
          <div className="flex items-center gap-4">
            <button className="relative p-2 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800">
              <FaBell />
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full px-1">3</span>
            </button>
            <button className="p-2 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800">
              <img src="/avatar.png" alt="User" className="w-7 h-7 rounded-full border border-zinc-300 dark:border-zinc-700" />
            </button>
          </div>
        </header>
        {/* Page Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-zinc-50 dark:bg-zinc-900">
          {children}
        </div>
      </main>
    </div>
  );
}
