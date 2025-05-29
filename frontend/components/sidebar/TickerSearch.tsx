import { useState } from 'react';
import { useTickerContext } from '@/context/TickerContext';

export default function TickerSearch() {
  const [input, setInput] = useState('');
  const { addTicker } = useTickerContext();

  const handleAdd = () => {
    if (input.trim()) {
      addTicker(input.toUpperCase());
      setInput('');
    }
  };

  return (
    <div className="flex gap-2 mt-2 font-sans">
      <input
        className="flex-1 px-3 py-1.5 rounded-lg bg-bgPanel text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accentBlue shadow border border-borderSoft"
        placeholder="Add ticker..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
      />
      <button
        className="bg-accentGreen text-black px-4 py-1.5 rounded-lg font-bold shadow-neon hover:shadow-glow transition"
        onClick={handleAdd}
      >
        Add
      </button>
    </div>
  );
}


