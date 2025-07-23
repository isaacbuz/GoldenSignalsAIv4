/**
 * Chart drawings hook
 * Manages drawing tools and annotations on the chart
 */

import { useState, useCallback } from 'react';

export interface Drawing {
  id: string;
  type: 'line' | 'trend' | 'horizontal' | 'rectangle' | 'circle' | 'fibonacci' | 'text' | 'pencil';
  points: { time: number; price: number }[];
  style?: {
    color?: string;
    lineWidth?: number;
    lineStyle?: 'solid' | 'dashed' | 'dotted';
    fillColor?: string;
    text?: string;
  };
  created: Date;
}

interface UseChartDrawingsReturn {
  drawings: Drawing[];
  addDrawing: (drawing: Omit<Drawing, 'id' | 'created'>) => void;
  removeDrawing: (id: string) => void;
  updateDrawing: (id: string, updates: Partial<Drawing>) => void;
  clearDrawings: () => void;
  saveDrawings: () => void;
  loadDrawings: (symbol: string) => void;
}

export function useChartDrawings(): UseChartDrawingsReturn {
  const [drawings, setDrawings] = useState<Drawing[]>([]);

  // Add a new drawing
  const addDrawing = useCallback((drawing: Omit<Drawing, 'id' | 'created'>) => {
    const newDrawing: Drawing = {
      ...drawing,
      id: `drawing-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      created: new Date(),
    };
    setDrawings(prev => [...prev, newDrawing]);
  }, []);

  // Remove a drawing
  const removeDrawing = useCallback((id: string) => {
    setDrawings(prev => prev.filter(d => d.id !== id));
  }, []);

  // Update a drawing
  const updateDrawing = useCallback((id: string, updates: Partial<Drawing>) => {
    setDrawings(prev =>
      prev.map(d => (d.id === id ? { ...d, ...updates } : d))
    );
  }, []);

  // Clear all drawings
  const clearDrawings = useCallback(() => {
    setDrawings([]);
  }, []);

  // Save drawings to localStorage
  const saveDrawings = useCallback(() => {
    const symbol = window.location.pathname.split('/').pop() || 'default';
    const key = `chart-drawings-${symbol}`;
    localStorage.setItem(key, JSON.stringify(drawings));
  }, [drawings]);

  // Load drawings from localStorage
  const loadDrawings = useCallback((symbol: string) => {
    const key = `chart-drawings-${symbol}`;
    const saved = localStorage.getItem(key);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setDrawings(parsed.map((d: any) => ({
          ...d,
          created: new Date(d.created),
        })));
      } catch (error) {
        console.error('Failed to load drawings:', error);
      }
    }
  }, []);

  return {
    drawings,
    addDrawing,
    removeDrawing,
    updateDrawing,
    clearDrawings,
    saveDrawings,
    loadDrawings,
  };
}
