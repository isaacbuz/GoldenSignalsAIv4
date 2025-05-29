/**
 * TrendCard Component
 *
 * Displays a motion-animated card summarizing a trend, with accessible markup and Tailwind CSS styling.
 * - Accessibility: Uses semantic HTML, ARIA roles/labels, and clear focus styles.
 * - Styling: All layout and colors use Tailwind CSS design tokens.
 * - Responsive: Layout adapts for mobile and desktop.
 * - Documentation: Clear docstring for maintainability.
 */
import { motion } from 'framer-motion';
import React from 'react';

interface TrendCardProps {
  trend: string;
  details?: string;
}

export default function TrendCard({ trend, details }: TrendCardProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      role="region"
      aria-label={`Trend: ${trend}`}
      tabIndex={0}
      className="bg-bgPanel p-5 md:p-6 rounded-xl shadow-neon font-sans border border-borderSoft text-white mb-2 animate-fadeIn focus:outline-none focus:ring-2 focus:ring-accentBlue"
    >
      <header className="mb-1">
        <h3 className="text-xl md:text-2xl font-bold text-accentBlue tracking-tight font-sans">
          {trend}
        </h3>
      </header>
      <div className="text-sm md:text-base text-gray-400">
        {details || 'Trend details here...'}
      </div>
    </motion.section>
  );
}
