import CentralChart from './components/CentralChart/CentralChart';
import FloatingOrb from './components/FloatingOrb/FloatingOrb';
// @ts-ignore
import OptionsChainTable from './components/OptionsChainTable/OptionsChainTable';
import PredictionTimeline from './components/PredictionTimeline/PredictionTimeline';
import SignalPanel from './components/SignalPanel/SignalPanel';
import * as sampleData from './utils/sampleData';
// Pass signals to a component if needed, e.g. <SignalPanel signals={signals} />

function App() {
    return (
        <div className="min-h-screen bg-dark-bg text-white flex flex-col">
            <header className="p-4 text-center text-golden-primary">GoldenSignalsAI Prophet</header>
            <main className="flex-grow flex justify-center items-center p-4">
                <div className="relative w-full max-w-7xl">
                    <CentralChart data={sampleData.sampleStockData} predictions={sampleData.samplePredictions} />
                    <FloatingOrb text="GoldenEye AI" />
                </div>
            </main>
            <aside className="flex">
                <SignalPanel />
                <OptionsChainTable />
            </aside>
            <footer className="p-4 text-center">
                <PredictionTimeline />
            </footer>
        </div>
    );
}

export default App; 