/**
 * ZoomControl Component
 *
 * Provides accessible controls for zooming in, out, and resetting a chart view.
 * - Accessibility: Uses semantic HTML, ARIA roles/labels, and keyboard navigation.
 * - Styling: All layout and colors use Tailwind CSS design tokens.
 * - Responsive: Layout adapts for mobile and desktop.
 * - Documentation: Clear docstring for maintainability.
 */
import React from 'react';

interface ZoomControlProps {
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onReset?: () => void;
  disabled?: boolean;
}

export default function ZoomControl({ onZoomIn, onZoomOut, onReset, disabled }: ZoomControlProps) {
  return (
    <nav
      aria-label="Chart zoom controls"
      className="flex items-center gap-2 md:gap-4 mt-2"
    >
      <button
        type="button"
        aria-label="Zoom out"
        onClick={onZoomOut}
        disabled={disabled}
        className="px-3 py-2 rounded-lg bg-bgPanel border border-borderSoft text-accentBlue hover:bg-accentBlue/10 focus:outline-none focus:ring-2 focus:ring-accentBlue disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <span aria-hidden="true" className="text-lg font-bold">-</span>
      </button>
      <button
        type="button"
        aria-label="Reset zoom"
        onClick={onReset}
        disabled={disabled}
        className="px-3 py-2 rounded-lg bg-bgPanel border border-borderSoft text-accentBlue hover:bg-accentBlue/10 focus:outline-none focus:ring-2 focus:ring-accentBlue disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <span aria-hidden="true" className="text-lg font-bold">⟳</span>
      </button>
      <button
        type="button"
        aria-label="Zoom in"
        onClick={onZoomIn}
        disabled={disabled}
        className="px-3 py-2 rounded-lg bg-bgPanel border border-borderSoft text-accentBlue hover:bg-accentBlue/10 focus:outline-none focus:ring-2 focus:ring-accentBlue disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <span aria-hidden="true" className="text-lg font-bold">+</span>
      </button>
    </nav>
  );
}
