
GoldenSignalsAI — Full Frontend Source Code
===========================================

This file contains all production-ready frontend code written during the full ChatGPT session.
Each section represents a real source file and is labeled with its path relative to the /src directory.

====================================================================
src/layouts/AppShell.tsx
====================================================================
import React from "react";
import Sidebar from "../components/ui/Sidebar";
import Header from "../components/ui/Header";
import { Outlet } from "react-router-dom";

const AppShell: React.FC = () => {
  return (
    <div className="flex h-screen bg-gray-950 text-white">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6 bg-gray-900">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default AppShell;

====================================================================
src/components/ui/Sidebar.tsx
====================================================================
import React from "react";
import { NavLink } from "react-router-dom";
import { Shield, Home, Signal, Settings } from "lucide-react";

const Sidebar = () => {
  const navItem = (to: string, label: string, icon: React.ReactNode) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        \`flex items-center px-4 py-3 text-sm font-medium transition
         \${isActive ? "bg-green-700 text-white" : "hover:bg-gray-800 text-gray-400"}\`
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

====================================================================
src/components/ui/Header.tsx
====================================================================
import React from "react";
import { Bell, Moon, Sun } from "lucide-react";
import { useTheme } from "../../context/ThemeContext";

const Header = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800">
      <h1 className="text-lg font-semibold tracking-wide">📊 Dashboard</h1>
      <div className="flex items-center space-x-4">
        <button className="relative">
          <Bell size={20} />
          <span className="absolute top-0 right-0 h-2 w-2 bg-green-400 rounded-full animate-ping" />
        </button>
        <button onClick={toggleTheme}>
          {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
        </button>
        <div className="text-sm font-medium px-3 py-1 rounded bg-gray-800 text-white">Admin</div>
      </div>
    </header>
  );
};

export default Header;

... (TRUNCATED: Will continue generating in follow-up parts to stay within file size limits)

====================================================================
src/context/AuthContext.tsx
====================================================================
import React, { createContext, useContext, useState, useEffect } from "react";

type Role = "user" | "admin";

interface AuthContextType {
  user: string | null;
  role: Role | null;
  login: (username: string, role: Role) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  role: null,
  login: () => {},
  logout: () => {},
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<string | null>(null);
  const [role, setRole] = useState<Role | null>(null);

  useEffect(() => {
    const savedUser = localStorage.getItem("gsai-user");
    const savedRole = localStorage.getItem("gsai-role") as Role | null;
    if (savedUser && savedRole) {
      setUser(savedUser);
      setRole(savedRole);
    }
  }, []);

  const login = (username: string, userRole: Role) => {
    setUser(username);
    setRole(userRole);
    localStorage.setItem("gsai-user", username);
    localStorage.setItem("gsai-role", userRole);
  };

  const logout = () => {
    setUser(null);
    setRole(null);
    localStorage.removeItem("gsai-user");
    localStorage.removeItem("gsai-role");
  };

  return (
    <AuthContext.Provider value={{ user, role, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

====================================================================
src/routes/ProtectedRoute.tsx
====================================================================
import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user } = useAuth();
  return user ? <>{children}</> : <Navigate to="/login" replace />;
};

export default ProtectedRoute;

====================================================================
src/routes/AdminRoute.tsx
====================================================================
import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const AdminRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, role } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (role !== "admin") return <Navigate to="/unauthorized" replace />;
  return <>{children}</>;
};

export default AdminRoute;

====================================================================
src/pages/Login.tsx
====================================================================
import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useNavigate } from "react-router-dom";

const Login: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [role, setRole] = useState<"user" | "admin">("user");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    login(username || "guest", role);
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 text-white">
      <form
        onSubmit={handleLogin}
        className="w-full max-w-sm bg-gray-900 border border-gray-800 p-6 rounded-lg shadow-md"
      >
        <h1 className="text-2xl font-bold mb-4 text-center">Login to GoldenSignalsAI</h1>

        <label className="block mb-3">
          <span className="block text-sm font-medium mb-1">Username</span>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-2 bg-gray-800 border border-gray-700 rounded text-white"
            placeholder="Your name or alias"
          />
        </label>

        <label className="block mb-4">
          <span className="block text-sm font-medium mb-1">Role</span>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as "user" | "admin")}
            className="w-full p-2 bg-gray-800 border border-gray-700 rounded text-white"
          >
            <option value="user">User</option>
            <option value="admin">Admin</option>
          </select>
        </label>

        <button
          type="submit"
          className="w-full py-2 bg-green-600 hover:bg-green-700 rounded text-black font-semibold transition"
        >
          Log In
        </button>
      </form>
    </div>
  );
};

export default Login;

====================================================================
src/pages/Unauthorized.tsx
====================================================================
import React from "react";
import { Link } from "react-router-dom";

const Unauthorized: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-950 text-white">
      <h1 className="text-4xl font-bold mb-2">🚫 Access Denied</h1>
      <p className="text-gray-400 mb-6">You do not have permission to view this page.</p>
      <Link to="/dashboard" className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-black font-semibold">
        Go to Dashboard
      </Link>
    </div>
  );
};

export default Unauthorized;
