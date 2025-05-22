import React from 'react';

export default function NotificationToast({ message, type = 'info', onClose }) {
  if (!message) return null;
  return (
    <div
      role="alert"
      aria-live="assertive"
      className={`notification-toast notification-toast-${type}`}
      style={{
        position: 'fixed',
        top: 24,
        right: 24,
        zIndex: 9999,
        background: type === 'error' ? '#ff3333' : type === 'success' ? '#00d084' : '#232323',
        color: '#fff',
        padding: '12px 24px',
        borderRadius: 8,
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        minWidth: 180
      }}
    >
      {message}
      <button aria-label="Close" onClick={onClose} style={{ marginLeft: 16, background: 'transparent', color: '#fff', border: 'none', fontWeight: 700, cursor: 'pointer' }}>
        ×
      </button>
    </div>
  );
}
