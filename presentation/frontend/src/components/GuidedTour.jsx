import React, { useState } from 'react';

const steps = [
  { target: '#live-signals', content: 'Here’s where your live signals appear in real time.' },
  { target: '#main-chart', content: 'This is your interactive chart. Hover/click markers for details.' },
  { target: '#quick-actions', content: 'Use quick actions to backtest, create orders, or add to watchlist.' },
  { target: '#settings', content: 'Customize your signal rules and notifications here.' },
];

export default function GuidedTour({ onFinish }) {
  const [step, setStep] = useState(0);

  if (step >= steps.length) {
    localStorage.setItem('hasSeenTour', '1');
    onFinish && onFinish();
    return null;
  }

  const { target, content } = steps[step];
  const targetEl = document.querySelector(target);
  const rect = targetEl ? targetEl.getBoundingClientRect() : { top: 100, left: 100, width: 200 };

  return (
    <div style={{
      position: 'fixed',
      top: rect.top + window.scrollY - 10,
      left: rect.left + window.scrollX + rect.width + 10,
      background: '#232323',
      color: '#FFD700',
      border: '2px solid #FFD700',
      borderRadius: 12,
      padding: 18,
      zIndex: 2000,
      width: 280,
      boxShadow: '0 2px 16px #FFD70055'
    }}>
      <div style={{ marginBottom: 12 }}>{content}</div>
      <button style={{ background: '#FFD700', color: '#232323', border: 'none', borderRadius: 6, padding: '6px 18px', fontWeight: 700 }}
        onClick={() => setStep(s => s + 1)}>
        {step === steps.length - 1 ? 'Finish' : 'Next'}
      </button>
    </div>
  );
}
