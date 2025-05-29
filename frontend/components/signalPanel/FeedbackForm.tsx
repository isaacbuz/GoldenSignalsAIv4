import React, { useState } from "react";

interface FeedbackFormProps {
  symbol: string;
  agent: string;
  onSubmitted?: () => void;
}

export default function FeedbackForm({ symbol, agent, onSubmitted }: FeedbackFormProps) {
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");

  const submitFeedback = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    setSuccess(false);
    try {
      const res = await fetch("/api/agents/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, agent, rating, comment }),
      });
      if (!res.ok) throw new Error("Failed to submit feedback");
      setSuccess(true);
      setRating(0);
      setComment("");
      if (onSubmitted) onSubmitted();
    } catch (e) {
      setError("Error submitting feedback");
    }
    setSubmitting(false);
  };

  return (
    <form className="feedback-form" onSubmit={submitFeedback}>
      <div>
        <label>Rating: </label>
        <select value={rating} onChange={e => setRating(Number(e.target.value))}>
          <option value={0}>Select</option>
          <option value={1}>1 (Bearish)</option>
          <option value={2}>2</option>
          <option value={3}>3 (Neutral)</option>
          <option value={4}>4</option>
          <option value={5}>5 (Bullish)</option>
        </select>
      </div>
      <div>
        <label>Comment: </label>
        <input value={comment} onChange={e => setComment(e.target.value)} />
      </div>
      <button type="submit" disabled={submitting}>Submit</button>
      {success && <span className="success">Thank you for your feedback!</span>}
      {error && <span className="error">{error}</span>}
    </form>
  );
}
