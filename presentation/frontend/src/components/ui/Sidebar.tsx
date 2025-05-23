
import React from "react";
import { NavLink } from "react-router-dom";
import { Shield, Home, Signal, Settings } from "lucide-react";

const Sidebar = () => {
  const navItem = (to: string, label: string, icon: React.ReactNode) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center px-4 py-3 text-sm font-medium transition
         ${isActive ? "bg-green-700 text-white" : "hover:bg-gray-800 text-gray-400"}`
      }
    >
      <span className="mr-2">{icon}</span> {label}
    </NavLink>
  );

  return (
    <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="text-2xl font-bold px-4 py-5">GoldenSignalsAI</div>
      <nav className="flex-1 flex flex-col gap-1">
        {navItem("/dashboard", "Dashboard", <Home size={18} />)}
        {navItem("/signals", "Signals", <Signal size={18} />)}
        {navItem("/admin", "Admin", <Shield size={18} />)}
        {navItem("/settings", "Settings", <Settings size={18} />)}
      </nav>
    </aside>
  );
};

export default Sidebar;
