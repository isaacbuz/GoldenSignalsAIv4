import React, { useState } from "react";

export default function UserPreferences() {
  const [theme, setTheme] = useState("system");
  const [notifications, setNotifications] = useState(true);

  return (
    <section className="p-4 bg-white dark:bg-zinc-900 rounded shadow mb-4">
      <h2 className="text-lg font-bold mb-2">User Preferences</h2>
      <div className="mb-2">
        <label className="mr-2">Theme:</label>
        <select value={theme} onChange={e => setTheme(e.target.value)} className="rounded p-1">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>
      <div className="mb-2">
        <label>
          <input type="checkbox" checked={notifications} onChange={e => setNotifications(e.target.checked)} className="mr-2" />
          Enable notifications
        </label>
      </div>
      <div className="text-zinc-500 text-xs">(Preferences are local only for now.)</div>
    </section>
  );
}
