'use client';
import { ReactNode } from 'react';

export default function ErrorBoundary({
  fallback,
  children
}: {
  fallback: ReactNode;
  children: ReactNode;
}) {
  return (
    <div>
      {/* Placeholder logic - Replace with real error handling */}
      {children}
    </div>
  );
}
