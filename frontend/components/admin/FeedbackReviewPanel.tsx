import React, { useState, useEffect } from "react";

function Toast({ message, type, onClose }: { message: string; type: 'success' | 'error'; onClose: () => void }) {
  return (
    <div style={{
      position: 'fixed', bottom: 32, right: 32, zIndex: 2000, background: type === 'success' ? '#4caf50' : '#d32f2f', color: 'white', padding: '16px 32px', borderRadius: 8, boxShadow: '0 4px 16px #0002', fontWeight: 600
    }}>
      {message}
      <button onClick={onClose} style={{ marginLeft: 16, background: 'none', color: 'white', border: 'none', fontWeight: 700, cursor: 'pointer' }}>×</button>
    </div>
  );
}

interface FeedbackEntry {
  id: string;
  symbol: string;
  agent: string;
  action: string;
  rating: number;
  comment: string;
  timestamp: string;
}

export default function FeedbackReviewPanel() {
  const [feedback, setFeedback] = useState<FeedbackEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState<{ message: string, type: 'success' | 'error' } | null>(null);

  const fetchFeedback = (showToast = false) => {
    setLoading(true);
    setError("");
    fetch("/api/agents/feedback")
      .then((res) => res.json())
      .then((data) => {
        setFeedback(data.feedback || []);
        setLoading(false);
        if (showToast) setToast({ message: "Feedback refreshed!", type: "success" });
      })
      .catch(() => {
        setError("Failed to load feedback.");
        setLoading(false);
        setToast({ message: "Failed to load feedback.", type: "error" });
      });
  };

  useEffect(() => {
    fetchFeedback(false);
  }, []);

  return (
    <div className="feedback-review-panel">
      <h2>Admin: Feedback Review</h2>
      <button onClick={() => fetchFeedback(true)} style={{ marginBottom: 8 }}>Refresh</button>
      {loading ? (
        <div>Loading...</div>
      ) : error ? (
        <div className="error">{error}</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Agent</th>
              <th>Action</th>
              <th>Rating</th>
              <th>Comment</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {feedback.map((f) => (
              <tr key={f.id}>
                <td>{f.symbol}</td>
                <td>{f.agent}</td>
                <td>{f.action}</td>
                <td>{f.rating}</td>
                <td>{f.comment}</td>
                <td>{f.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    {toast && (
      <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
    )}
  </div>
  );
}
