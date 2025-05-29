import { useState } from 'react';

export default function AdminPanel() {
  const [rules, setRules] = useState<any>({});
  const [json, setJson] = useState('');

  const handleLoad = async () => {
    const res = await fetch('/api/admin/signal-rules');
    const data = await res.json();
    setRules(data);
    setJson(JSON.stringify(data, null, 2));
  };

  const handleSave = async () => {
    await fetch('/api/admin/signal-rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: json,
    });
    alert('Saved!');
  };

  return (
    <div className="p-6 text-white bg-[#0f172a] min-h-screen">
      <h1 className="text-2xl font-bold mb-4 text-neonPink">Admin: Signal Logic</h1>
      <textarea
        value={json}
        onChange={(e) => setJson(e.target.value)}
        className="w-full h-[400px] bg-black text-green-300 p-4 font-mono rounded"
      />
      <div className="flex gap-4 mt-4">
        <button onClick={handleLoad} className="bg-neonGreen text-black px-4 py-2 rounded">Load</button>
        <button onClick={handleSave} className="bg-neonBlue text-black px-4 py-2 rounded">Save</button>
      </div>
    </div>
  );
}
