import SignalSummaryCard from './SignalSummaryCard';
import RadarChart from './RadarChart';
import AutoTradeToggle from './AutoTradeToggle';

import AgentConsensusPanel from './AgentConsensusPanel';

export default function SignalPanel() {
  // Symbol is now dynamic via props or context
  const symbol = 'AAPL';
  return (
    <div className="space-y-4">
      <SignalSummaryCard />
      <RadarChart />
      <AutoTradeToggle />
      <AgentConsensusPanel symbol={symbol} />
    </div>
  );
}


