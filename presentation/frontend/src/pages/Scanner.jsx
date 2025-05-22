import React, { useState, useEffect } from 'react';
import GrokFeedback from '../components/GrokFeedback';

function Scanner() {
  // === Mock Grok feedback for demonstration ===
  const [grokFeedback, setGrokFeedback] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFeedback();
    // eslint-disable-next-line
  }, []);

  const fetchFeedback = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/grok/feedback');
      if (!res.ok) throw new Error('Failed to fetch Grok feedback');
      const data = await res.json();
      setGrokFeedback(data.suggestions || []);
    } catch (err) {
      setError(err.message);
      setGrokFeedback([]);
    }
    setLoading(false);
  };


  return (
    <div className="scanner-page">
      <h1>Market Scanner</h1>
      <p>Find the top 5 options trades with the highest probability of profit based on current market analysis.</p>
      {/* === Grok AI Feedback Section === */}
      <button onClick={fetchFeedback} disabled={loading} style={{marginBottom: 12}}>
        {loading ? 'Analyzing...' : 'Request Grok Critique'}
      </button>
      {error && <div style={{color: 'red', marginBottom: 8}}>Error: {error}</div>}
      <GrokFeedback feedback={grokFeedback} loading={loading} />
      {/* Scanner logic and results table will go here */}
      {/* === Placeholder for strategy logic and Grok integration === */}
      {/* Example: Display strategy logic and allow user to request Grok critique */}
    </div>
  );
}

export default Scanner;
