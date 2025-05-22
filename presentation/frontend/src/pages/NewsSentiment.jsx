import React, { useState } from "react";
import { fetchNewsSentiment } from "../api/index";

export default function NewsSentiment() {
  const [topic, setTopic] = useState("TSLA");
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFetch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetchNewsSentiment(topic);
      setNews(res.headlines || []);
    } catch (err) {
      setError(err.message || "Error fetching news");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="news-sentiment-container">
      <h2>News & Sentiment Feed</h2>
      <form onSubmit={handleFetch} style={{ marginBottom: 16 }}>
        <input
          type="text"
          value={topic}
          onChange={e => setTopic(e.target.value)}
          placeholder="Enter topic or ticker..."
        />
        <button type="submit" disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? "Loading..." : "Fetch News"}
        </button>
      </form>
      {error && <div style={{ color: "red" }}>{error}</div>}
      <ul style={{ listStyle: "none", padding: 0 }}>
        {news.map((item, idx) => (
          <li key={idx} style={{ marginBottom: 12, padding: 12, border: "1px solid #eee", borderRadius: 8 }}>
            <strong>{item.headline}</strong>
            <div>
              <span style={{ color: item.sentiment === "positive" ? "green" : item.sentiment === "negative" ? "red" : "#888" }}>
                {item.sentiment} ({item.score})
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
