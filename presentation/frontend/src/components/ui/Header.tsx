
import React from "react";
import { Bell, Moon, Sun } from "lucide-react";


const Header = () => {
  return (
    <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800">
      <h1 className="text-lg font-semibold tracking-wide">📊 Dashboard</h1>
      <div className="flex items-center space-x-4">
        <button className="relative">
          <Bell size={20} />
          <span className="absolute top-0 right-0 h-2 w-2 bg-green-400 rounded-full animate-ping" />
        </button>
        <div className="text-sm font-medium px-3 py-1 rounded bg-gray-800 text-white">Admin</div>
      </div>
    </header>
  );
};

export default Header;
