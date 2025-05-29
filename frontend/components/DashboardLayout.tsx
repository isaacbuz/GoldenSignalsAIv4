import React from "react";
import { Home, BarChart2, List, RefreshCcw, Settings } from "lucide-react";
import Link from "next/link";
import ThemeToggle from "./ThemeToggle";

const nav = [
  { href: "/", label: "Home", icon: <Home size={18} /> },
  { href: "/analytics", label: "Analytics", icon: <BarChart2 size={18} /> },
  { href: "/logs", label: "Logs", icon: <List size={18} /> },
  { href: "/replay", label: "Replay", icon: <RefreshCcw size={18} /> },
  { href: "/settings", label: "Settings", icon: <Settings size={18} /> },
];

/**
 * DashboardLayout provides a fintech-style shell with sidebar navigation, theme toggle, and main content area.
 * Uses Tailwind tokens, Inter font, and accessibility best practices.
 */

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  // Get current path for aria-current (active link)
  const currentPath = typeof window !== 'undefined' ? window.location.pathname : '';
  return (
    <div className="flex min-h-screen bg-bgPanel text-white font-sans">
      {/* Sidebar */}
      <aside className="w-56 bg-bgDark border-r border-borderSoft flex flex-col py-6 px-4 gap-2 rounded-r-xl shadow-neon">
        <div className="text-2xl font-bold mb-8 tracking-tight text-accentBlue font-sans">GoldenSignalsAI</div>
        <nav className="flex flex-col gap-2" aria-label="Sidebar navigation">
          {nav.map(({ href, label, icon }) => (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-borderSoft transition-colors font-sans ${currentPath === href ? 'bg-borderSoft text-accentGreen' : ''}`}
              aria-current={currentPath === href ? 'page' : undefined}
            >
              {icon}
              <span className="text-base">{label}</span>
            </Link>
          ))}
        </nav>
        <div className="mt-4">
          {/* Theme toggle */}
          <div className="flex justify-center"><ThemeToggle /></div>
        </div>
        <div className="mt-auto text-xs text-accentBlue/60">Fintech Dashboard &copy; 2025</div>
      </aside>
      {/* Main content */}
      <main className="flex-1 p-8 md:p-12 bg-bgPanel min-h-screen overflow-y-auto font-sans">
        {children}
      </main>
    </div>
  );
}
