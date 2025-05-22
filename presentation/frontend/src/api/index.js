// API Adapter for GoldenSignalsAI backend
// Handles all /api/v1/ endpoints

const API_URL = import.meta.env.VITE_API_URL || '';
if (!API_URL) {
  // eslint-disable-next-line no-console
  console.error('VITE_API_URL is not defined in your environment variables.');
}
const API_BASE = API_URL || "/api/v1";

export async function fetchNewsSentiment(topic = "TSLA", max_articles = 10) {
  const res = await fetch(`${API_BASE}/news_agent/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Add auth headers or JWT if needed
    },
    body: JSON.stringify({ topic, max_articles }),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function fetchWatchlist(user_id) {
  const res = await fetch(`${API_BASE}/watchlist/get/${user_id}`);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

// Add more API adapters as needed
