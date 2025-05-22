import React from 'react';

export default function HeatmapOverview({ data }) {
  // data: [{ symbol, buyCount, sellCount }]
  const maxBuy = Math.max(...data.map(d => d.buyCount));
  const maxSell = Math.max(...data.map(d => d.sellCount));
  return (
    <div style={{ display: 'flex', gap: 4, margin: '8px 0', flexWrap: 'wrap' }}>
      {data.map((d, i) => {
        const buyRatio = d.buyCount / (maxBuy || 1);
        const sellRatio = d.sellCount / (maxSell || 1);
        const bg = buyRatio > sellRatio
          ? `rgba(0,255,100,${0.3 + 0.5 * buyRatio})`
          : `rgba(255,60,60,${0.3 + 0.5 * sellRatio})`;
        return (
          <div key={d.symbol} style={{
            width: 36, height: 36, borderRadius: 8, background: bg,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#FFD700', fontWeight: 700, fontSize: 13, boxShadow: '0 1px 8px #FFD70022', marginBottom: 4
          }}>
            <div>{d.symbol}</div>
            <div style={{ fontSize: 11 }}>{d.buyCount + d.sellCount}</div>
          </div>
        );
      })}
    </div>
  );
}
