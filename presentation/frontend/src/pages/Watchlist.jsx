import React, { useState } from "react";
import { fetchWatchlist } from "../api/index";

export default function Watchlist() {
  const [userId, setUserId] = useState("");
  const [watchlist, setWatchlist] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFetch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetchWatchlist(userId);
      setWatchlist(res.watchlist || []);
    } catch (err) {
      setError(err.message || "Error fetching watchlist");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="watchlist-container">
      <h2>Watchlist</h2>
      <form onSubmit={handleFetch} style={{ marginBottom: 16 }}>
        <input
          type="text"
          value={userId}
          onChange={e => setUserId(e.target.value)}
          placeholder="Enter User ID..."
        />
        <button type="submit" disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? "Loading..." : "Fetch Watchlist"}
        </button>
      </form>
      {error && <div style={{ color: "red" }}>{error}</div>}
      <ul style={{ listStyle: "none", padding: 0 }}>
        {watchlist.map((item, idx) => (
          <li key={idx} style={{ marginBottom: 12, padding: 12, border: "1px solid #eee", borderRadius: 8 }}>
            <strong>{item.ticker}</strong>
            {item.tags && item.tags.length > 0 && (
              <span style={{ marginLeft: 8, color: "#888" }}>
                [{item.tags.join(", ")}]
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
