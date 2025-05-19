import React, { useState } from 'react';
import './TradeJournal.css';

// Demo/mock data for journal entries
const MOCK_ENTRIES = [
  {
    id: 1,
    symbol: 'AAPL',
    date: '2025-05-17',
    side: 'Buy',
    qty: 10,
    price: 175.2,
    result: '+$320',
    tags: ['breakout'],
    notes: 'Strong volume on breakout. Good entry.'
  },
  {
    id: 2,
    symbol: 'TSLA',
    date: '2025-05-14',
    side: 'Sell',
    qty: 5,
    price: 728.1,
    result: '-$85',
    tags: ['reversal'],
    notes: 'Premature exit, should have waited for confirmation.'
  },
];

export default function TradeJournal() {
  const [entries, setEntries] = useState(MOCK_ENTRIES);
  const [filter, setFilter] = useState('');
  const [newNote, setNewNote] = useState('');
  const [selectedId, setSelectedId] = useState(null);

  const filtered = entries.filter(e =>
    (!filter || e.symbol.toLowerCase().includes(filter.toLowerCase()))
  );

  function handleAddNote() {
    if (!newNote.trim() || selectedId == null) return;
    setEntries(entries.map(e =>
      e.id === selectedId ? { ...e, notes: e.notes + '\n' + newNote } : e
    ));
    setNewNote('');
  }

  return (
    <div className="trade-journal-panel">
      <div className="journal-header">
        <h3>Trade Journal</h3>
        <input
          className="journal-filter"
          placeholder="Filter by symbol..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
      </div>
      <div className="journal-entries">
        {filtered.map(e => (
          <div key={e.id} className={`journal-entry${selectedId === e.id ? ' selected' : ''}`}
               onClick={() => setSelectedId(e.id)}>
            <div className="entry-main">
              <span className="entry-symbol">{e.symbol}</span>
              <span className="entry-date">{e.date}</span>
              <span className={`entry-side ${e.side.toLowerCase()}`}>{e.side}</span>
              <span className="entry-qty">{e.qty} @ ${e.price}</span>
              <span className={`entry-result ${e.result.startsWith('+') ? 'profit' : 'loss'}`}>{e.result}</span>
              <span className="entry-tags">{e.tags.map(t => <span key={t} className="tag">{t}</span>)}</span>
            </div>
            <div className="entry-notes">{e.notes}</div>
            {selectedId === e.id && (
              <div className="entry-add-note">
                <textarea
                  value={newNote}
                  onChange={e => setNewNote(e.target.value)}
                  placeholder="Add note..."
                  rows={2}
                />
                <button onClick={handleAddNote}>Add Note</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
