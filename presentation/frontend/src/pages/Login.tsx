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
