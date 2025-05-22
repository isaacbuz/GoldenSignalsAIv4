import React, { useState, useRef, useEffect } from 'react';

// Uses Financial Modeling Prep API for symbol search
// Set REACT_APP_FMP_API_KEY in your .env file (see https://financialmodelingprep.com/developer/docs/)
const API_URL = 'https://financialmodelingprep.com/api/v3/search';
const API_KEY = import.meta.env.VITE_FMP_API_KEY || '';
if (!API_KEY) {
  // eslint-disable-next-line no-console
  console.warn('VITE_FMP_API_KEY is not defined in your environment variables. Symbol autocomplete may not work.');
}



export default function TickerAutocomplete({ value, onChange, onSelect, ...props }) {
  const [input, setInput] = useState(value || '');
  const [suggestions, setSuggestions] = useState([]);
  const [show, setShow] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    setInput(value || '');
  }, [value]);

  useEffect(() => {
    const handler = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setShow(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Debounced fetch for symbol autocomplete
  useEffect(() => {
    if (!input || input.length < 1) {
      setSuggestions([]);
      setShow(false);
      return;
    }
    const controller = new AbortController();
    const timeout = setTimeout(() => {
      fetch(`${API_URL}?query=${input}&limit=8&exchange=NASDAQ${API_KEY ? `&apikey=${API_KEY}` : ''}`,
        { signal: controller.signal })
        .then(res => res.json())
        .then(data => {
          setSuggestions(data || []);
          setShow((data || []).length > 0);
        })
        .catch(() => setSuggestions([]));
    }, 250);
    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [input]);

  function handleInput(e) {
    const val = e.target.value.toUpperCase();
    setInput(val);
    onChange(val);
  }

  function handleSelect(sym) {
    setInput(sym.symbol);
    setShow(false);
    onChange(sym.symbol);
    if (onSelect) onSelect(sym.symbol);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && suggestions.length > 0) {
      handleSelect(suggestions[0]);
    }
  }

  return (
    <div className="ticker-autocomplete" ref={containerRef} style={{ position: 'relative', width: '110px' }}>
      <input
        {...props}
        type="text"
        value={input}
        onChange={handleInput}
        onFocus={() => { if (suggestions.length > 0) setShow(true); }}
        onKeyDown={handleKeyDown}
        autoComplete="off"
        aria-label="Enter ticker symbol"
        style={{ width: '100%', padding: '6px 8px', border: '1px solid #e5e7eb', borderRadius: 6, fontSize: '1rem', marginRight: 8 }}
      />
      {show && suggestions.length > 0 && (
        <ul className="ticker-suggestions" style={{
          position: 'absolute',
          top: '110%',
          left: 0,
          zIndex: 10,
          background: '#fff',
          border: '1px solid #e5e7eb',
          borderRadius: 6,
          boxShadow: '0 2px 8px #FFD70022',
          listStyle: 'none',
          margin: 0,
          padding: '0.2em 0',
          width: '100%',
          fontSize: '1rem',
          color: '#232323',
        }}>
          {suggestions.map(sym => (
            <li
              key={sym.symbol}
              onMouseDown={() => handleSelect(sym)}
              style={{ padding: '0.35em 0.7em', cursor: 'pointer' }}
              tabIndex={0}
              onKeyDown={e => { if (e.key === 'Enter') handleSelect(sym); }}
            >
              <span style={{ fontWeight: 700 }}>{sym.symbol}</span>{' '}
              <span style={{ color: '#888', fontSize: '0.94em' }}>{sym.name}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
