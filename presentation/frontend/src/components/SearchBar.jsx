import React, { useState } from "react";

export default function SearchBar({ onSearch }) {
  const [symbol, setSymbol] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (symbol.trim()) onSearch(symbol.trim().toUpperCase());
  };

  return (
    <form onSubmit={handleSubmit} className="flex space-x-2 mb-6">
      <input
        type="text"
        placeholder="Enter stock symbol (e.g. AAPL)"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        className="p-2 border rounded w-full"
      />
      <button
        type="submit"
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Search
      </button>
    </form>
  );
}
