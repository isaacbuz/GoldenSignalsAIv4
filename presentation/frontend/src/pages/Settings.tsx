import React, { useEffect, useState } from "react";

const DEFAULTS = {
  rsiOverbought: 70,
  rsiOversold: 30,
  macdThreshold: 0.03,
  notifyPush: true,
  notifySound: true,
  muteAfterHours: false,
};

const Settings = () => {
  const [settings, setSettings] = useState(DEFAULTS);

  useEffect(() => {
    const saved = localStorage.getItem("gsai-settings");
    if (saved) setSettings(JSON.parse(saved));
  }, []);

  const save = () => {
    localStorage.setItem("gsai-settings", JSON.stringify(settings));
    alert("Settings saved.");
  };

  const reset = () => setSettings(DEFAULTS);

  const update = (key: keyof typeof settings, val: any) =>
    setSettings((s) => ({ ...s, [key]: val }));

  return (
    <div className="max-w-xl mx-auto p-6 text-white bg-gray-900 rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4">⚙️ Strategy Settings</h2>

      <div className="mb-4">
        <h3 className="font-semibold">RSI Thresholds</h3>
        <div className="flex gap-4 mt-2">
          <input
            type="number"
            value={settings.rsiOversold}
            onChange={(e) => update("rsiOversold", +e.target.value)}
            className="p-2 rounded bg-gray-800 w-1/2"
            placeholder="Oversold"
          />
          <input
            type="number"
            value={settings.rsiOverbought}
            onChange={(e) => update("rsiOverbought", +e.target.value)}
            className="p-2 rounded bg-gray-800 w-1/2"
            placeholder="Overbought"
          />
        </div>
      </div>

      <div className="mb-4">
        <h3 className="font-semibold">MACD Trigger</h3>
        <input
          type="number"
          value={settings.macdThreshold}
          onChange={(e) => update("macdThreshold", +e.target.value)}
          className="p-2 rounded bg-gray-800 w-full mt-2"
        />
      </div>

      <div className="mb-4">
        <h3 className="font-semibold">Notifications</h3>
        <label className="block mt-2">
          <input
            type="checkbox"
            checked={settings.notifyPush}
            onChange={(e) => update("notifyPush", e.target.checked)}
          />{" "}
          Browser Push
        </label>
        <label className="block">
          <input
            type="checkbox"
            checked={settings.notifySound}
            onChange={(e) => update("notifySound", e.target.checked)}
          />{" "}
          Sound Alerts
        </label>
        <label className="block">
          <input
            type="checkbox"
            checked={settings.muteAfterHours}
            onChange={(e) => update("muteAfterHours", e.target.checked)}
          />{" "}
          Mute After-Hours
        </label>
      </div>

      <div className="flex justify-between mt-6">
        <button
          onClick={reset}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-white"
        >
          Reset
        </button>
        <button
          onClick={save}
          className="px-4 py-2 bg-green-500 hover:bg-green-600 rounded text-black font-semibold"
        >
          Save Changes
        </button>
      </div>
    </div>
  );
};

export default Settings;
