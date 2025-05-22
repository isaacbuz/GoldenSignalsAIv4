import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './WatchlistSidebar.css';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

/**
 * WatchlistSidebar displays a list of tracked symbols with live price and mini-charts.
 * @param {Array} watchlist - Array of { symbol, price, change, sparkline }
 * @param {Function} onSelect - Function to call when a symbol is clicked
 */
import { motion, AnimatePresence } from 'framer-motion';

export default function WatchlistSidebar({ watchlist, selected, onSelect }) {
  const [symbols, setSymbols] = useState(watchlist);

  const handleDragEnd = (result) => {
    if (!result.destination) return;
    const items = Array.from(symbols);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    setSymbols(items);
  };

  return (
    <aside className="watchlist-sidebar card">
      <h3>Watchlist</h3>
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="watchlist">
          {(provided) => (
            <ul {...provided.droppableProps} ref={provided.innerRef}>
              <AnimatePresence>
                {symbols.map((symbolObj, index) => (
                  <Draggable key={symbolObj.symbol} draggableId={symbolObj.symbol} index={index}>
                    {(provided) => (
                      <motion.li
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                        className={`watchlist-item ${selected === symbolObj.symbol ? 'selected' : ''}`}
                        onClick={() => onSelect(symbolObj.symbol)}
                        onKeyDown={(e) => e.key === 'Enter' && onSelect(symbolObj.symbol)}
                        tabIndex={0}
                        role="button"
                        aria-label={`Select ${symbolObj.symbol}`}
                        initial={{ opacity: 0, y: 24 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 24 }}
                        transition={{ duration: 0.38, delay: index * 0.06, type: 'spring', stiffness: 70 }}
                        layout
                      >
                        <div
                          className="watchlist-symbol"
                          title={{
                            AAPL: 'Apple Inc.',
                            TSLA: 'Tesla, Inc.',
                            GE: 'General Electric',
                            MSFT: 'Microsoft Corp.'
                          }[symbolObj.symbol] || symbolObj.symbol}
                        >
                          {symbolObj.symbol}
                        </div>
                        <div className="watchlist-price">
                          <span className={symbolObj.change > 0 ? 'up' : symbolObj.change < 0 ? 'down' : ''}>
                            {symbolObj.price}
                          </span>
                          <span className="watchlist-change">
                            {symbolObj.change > 0 ? '+' : ''}{symbolObj.change}%
                          </span>
                        </div>
                        <div className="watchlist-sparkline">
                          {symbolObj.sparkline && (
                            <svg width="44" height="18" viewBox="0 0 44 18" style={{ marginLeft: 4 }} aria-label={`Mini chart for ${symbolObj.symbol}`}>
                              <polyline
                                fill="none"
                                stroke="#FFD700"
                                strokeWidth="2"
                                points={(() => {
                                  const values = symbolObj.sparkline.split(',').map(Number);
                                  const min = Math.min(...values), max = Math.max(...values);
                                  const norm = v => 16 - ((v - min) / (max - min + 0.01)) * 14;
                                  return values.map((v, i) => `${i * (42/(values.length-1))},${norm(v).toFixed(2)}`).join(' ');
                                })()}
                              />
                            </svg>
                          )}
                        </div>
                      </motion.li>
                    )}
                  </Draggable>
                ))}
              </AnimatePresence>
              {provided.placeholder}
            </ul>
          )}
        </Droppable>
      </DragDropContext>
    </aside>
  );
}

WatchlistSidebar.propTypes = {
  watchlist: PropTypes.arrayOf(PropTypes.shape({
    symbol: PropTypes.string.isRequired,
    price: PropTypes.number.isRequired,
    change: PropTypes.number.isRequired,
    sparkline: PropTypes.string.isRequired,
  })).isRequired,
  selected: PropTypes.string.isRequired,
  onSelect: PropTypes.func.isRequired,
};
