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
