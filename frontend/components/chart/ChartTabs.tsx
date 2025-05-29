/**
 * ChartTabs Component
 *
 * Provides an accessible, keyboard-navigable tab interface for switching between different chart types.
 * - Accessibility: Uses semantic HTML, ARIA roles/labels, and keyboard navigation.
 * - Styling: All layout and colors use Tailwind CSS design tokens.
 * - Responsive: Layout adapts for mobile and desktop.
 * - Documentation: Clear docstring for maintainability.
 */
import React, { useState, KeyboardEvent } from 'react';

const TABS = [
  { label: 'Price', value: 'price' },
  { label: 'Volume', value: 'volume' },
  { label: 'RSI', value: 'rsi' },
];

interface ChartTabsProps {
  onTabChange?: (tab: string) => void;
  initialTab?: string;
}

export default function ChartTabs({ onTabChange, initialTab }: ChartTabsProps) {
  const [activeTab, setActiveTab] = useState(initialTab || TABS[0].value);

  const handleTabClick = (tab: string) => {
    setActiveTab(tab);
    if (onTabChange) onTabChange(tab);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
    if (e.key === 'ArrowRight') {
      const next = (idx + 1) % TABS.length;
      setActiveTab(TABS[next].value);
      if (onTabChange) onTabChange(TABS[next].value);
    } else if (e.key === 'ArrowLeft') {
      const prev = (idx - 1 + TABS.length) % TABS.length;
      setActiveTab(TABS[prev].value);
      if (onTabChange) onTabChange(TABS[prev].value);
    }
  };

  return (
    <nav
      className="w-full mb-4"
      aria-label="Chart Type Tabs"
    >
      <ul className="flex flex-wrap gap-2 md:gap-4 justify-start" role="tablist">
        {TABS.map((tab, idx) => (
          <li key={tab.value} role="presentation">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === tab.value}
              aria-controls={`tabpanel-${tab.value}`}
              id={`tab-${tab.value}`}
              tabIndex={activeTab === tab.value ? 0 : -1}
              onClick={() => handleTabClick(tab.value)}
              onKeyDown={e => handleKeyDown(e, idx)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-accentBlue
                ${activeTab === tab.value
                  ? 'bg-accentBlue text-white shadow-md'
                  : 'bg-bgPanel text-accentBlue border border-borderSoft hover:bg-accentBlue/10'}
              `}
            >
              {tab.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
