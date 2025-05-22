import React, { useState, useEffect } from 'react';
import NotificationToast from './NotificationToast';

const API_URL = import.meta.env.VITE_API_URL || '';
const apiUrlWarning = !API_URL;
if (apiUrlWarning) {
  // eslint-disable-next-line no-console
  console.warn('VITE_API_URL is not defined in your environment variables. Some features may not work.');
}

function validate(settings) {
  const errors = {};
  if (settings.email && !/^\S+@\S+\.\S+$/.test(settings.email)) errors.email = 'Invalid email.';
  if (settings.slack && !/^https?:\/\//.test(settings.slack)) errors.slack = 'Invalid Slack webhook URL.';
  if (settings.discord && !/^https?:\/\//.test(settings.discord)) errors.discord = 'Invalid Discord webhook URL.';
  if (settings.sms && !/^\+?\d{7,}$/.test(settings.sms)) errors.sms = 'Invalid phone number.';
  return errors;
}

export default function NotificationSettings() {
  const [settings, setSettings] = useState({
    slack: '',
    discord: '',
    email: '',
    push: false,
    sms: '',
    highConfidenceOnly: true
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState({ message: '', type: 'info' });
  const [errors, setErrors] = useState({});

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setToast({ message: 'You must be logged in to manage notifications.', type: 'error' });
      setLoading(false);
      return;
    }
    fetch(`${API_URL}/user/notifications`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => {
        if (!res.ok) throw new Error('Failed to load settings');
        return res.json();
      })
      .then(data => {
        setSettings(prev => ({ ...prev, ...data }));
        setLoading(false);
      })
      .catch(() => {
        setToast({ message: 'Failed to load notification settings.', type: 'error' });
        setLoading(false);
      });
  }, []);

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  }

  function handleSave() {
    const validation = validate(settings);
    setErrors(validation);
    if (Object.keys(validation).length > 0) {
      setToast({ message: 'Please fix validation errors.', type: 'error' });
      return;
    }
    setSaving(true);
    const token = localStorage.getItem('token');
    fetch(`${API_URL}/user/notifications`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(settings)
    })
      .then(res => {
        if (!res.ok) throw new Error('Failed to save');
        setToast({ message: 'Notification settings saved!', type: 'success' });
      })
      .catch(() => setToast({ message: 'Failed to save settings.', type: 'error' }))
      .finally(() => setSaving(false));
  }

  return (
    <div className="notification-settings card" style={{ padding: 24, borderRadius: 10, background: '#232323', color: '#FFD700', maxWidth: 440 }}>
      {apiUrlWarning && (
        <div style={{ background: '#ff3333', color: '#fff', padding: '8px 12px', borderRadius: 6, marginBottom: 12, fontWeight: 600 }}>
          Warning: VITE_API_URL is not set. Notification settings will not work until this is configured.
        </div>
      )}
      <NotificationToast message={toast.message} type={toast.type} onClose={() => setToast({ message: '', type: 'info' })} />
      <h3 tabIndex={0}>Notification Settings</h3>
      {loading ? <div role="status" aria-live="polite">Loading...</div> : (
        <form onSubmit={e => { e.preventDefault(); handleSave(); }}>
          <label>Slack Webhook URL:
            <input name="slack" value={settings.slack} onChange={handleChange} style={{ width: '100%' }} aria-label="Slack Webhook URL" aria-invalid={!!errors.slack} />
            {errors.slack && <span style={{ color: 'red' }}>{errors.slack}</span>}
          </label>
          <label>Discord Webhook URL:
            <input name="discord" value={settings.discord} onChange={handleChange} style={{ width: '100%' }} aria-label="Discord Webhook URL" aria-invalid={!!errors.discord} />
            {errors.discord && <span style={{ color: 'red' }}>{errors.discord}</span>}
          </label>
          <label>Email:
            <input name="email" value={settings.email} onChange={handleChange} style={{ width: '100%' }} aria-label="Email" aria-invalid={!!errors.email} />
            {errors.email && <span style={{ color: 'red' }}>{errors.email}</span>}
          </label>
          <label>SMS Number:
            <input name="sms" value={settings.sms} onChange={handleChange} style={{ width: '100%' }} aria-label="SMS Number" aria-invalid={!!errors.sms} />
            {errors.sms && <span style={{ color: 'red' }}>{errors.sms}</span>}
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" name="push" checked={settings.push} onChange={handleChange} aria-label="Enable Push Notifications" />
            Enable Push Notifications
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" name="highConfidenceOnly" checked={settings.highConfidenceOnly} onChange={handleChange} aria-label="Only notify for high-confidence signals" />
            Only notify for high-confidence signals
          </label>
          <button type="submit" disabled={saving} style={{ marginTop: 18, background: '#FFD700', color: '#232323', border: 'none', borderRadius: 8, padding: '8px 18px', fontWeight: 700, opacity: saving ? 0.7 : 1 }}>
            {saving ? 'Saving...' : 'Save'}
          </button>
        </form>
      )}
    </div>
  );
}
