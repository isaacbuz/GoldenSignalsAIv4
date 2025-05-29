import React, { useEffect, useState, Suspense } from 'react';
import PerformanceAnalytics from './PerformanceAnalytics';
import ConfidenceHeatmap from './ui/ConfidenceHeatmap';
const TVComparisonCard = React.lazy(() => import('./ui/TVComparisonCard'));

export default function AnalyticsSection() {
  const [heatmap, setHeatmap] = useState<any[]>([]);
  const [tvData, setTvData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch('/api/confidence-heatmap').then(res => res.json()),
      fetch('/api/tv-comparison').then(res => res.json()),
    ]).then(([heatmapRes, tvRes]) => {
      setHeatmap(heatmapRes.data || []);
      setTvData(tvRes.data || null);
    }).finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-4">
      <PerformanceAnalytics />
      {loading ? (
        <div className="text-zinc-500">Loading analytics...</div>
      ) : (
        <>
          <ConfidenceHeatmap data={heatmap} />
          {tvData && (
            <Suspense fallback={<div>Loading...</div>}>
              <TVComparisonCard {...tvData} />
            </Suspense>
          )}
        </>
      )}
    </div>
  );
}
