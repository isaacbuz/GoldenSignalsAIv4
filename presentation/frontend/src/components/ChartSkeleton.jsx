import React from 'react';

export default function ChartSkeleton() {
  return (
    <div style={{
      width: '100%',
      maxWidth: 700,
      height: 380,
      margin: '0 auto',
      borderRadius: 16,
      background: 'linear-gradient(120deg, #232323 80%, #181818 100%)',
      boxShadow: '0 2px 12px #FFD70033',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <div className="skeleton-animation" style={{
        width: '90%',
        height: '70%',
        borderRadius: 12,
        background: 'linear-gradient(90deg, #232323 25%, #333 50%, #232323 75%)',
        backgroundSize: '200% 100%',
        animation: 'skeleton-loading 1.5s infinite linear'
      }} />
      <style>{`
        @keyframes skeleton-loading {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}
