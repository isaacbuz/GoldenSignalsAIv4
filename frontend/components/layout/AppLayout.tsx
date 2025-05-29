import Sidebar from '../sidebar/Sidebar';
import ChartPanel from '../chart/ChartPanel';
import SignalPanel from '../signalPanel/SignalPanel';

export default function AppLayout() {
  return (
    <div className="flex min-h-screen bg-bgDark text-white">
      <Sidebar />
      <main className="flex-1 p-4 grid grid-cols-3 gap-4">
        <section className="col-span-2 space-y-4">
          <ChartPanel />
        </section>
        <aside className="col-span-1 space-y-4">
          <SignalPanel />
        </aside>
      </main>
    </div>
  );
}

