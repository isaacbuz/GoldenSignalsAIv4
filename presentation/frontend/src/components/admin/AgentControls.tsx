import React from "react";
import ConfirmButton from "../ui/ConfirmButton";

interface Agent {
  id: string;
  name: string;
  enabled: boolean;
}

interface Props {
  agents: Agent[];
  onToggle: (id: string) => void;
  onRestart: (id: string) => void;
  onPurge: () => void;
}

const AgentControls: React.FC<Props> = ({ agents, onToggle, onRestart, onPurge }) => {
  return (
    <div className="mt-10">
      <h2 className="text-lg font-bold mb-4">🛠 Agent Control Center</h2>
      <ConfirmButton
        title="Purge All Signals"
        description="This will clear all system signals!"
        onConfirm={onPurge}
        className="mb-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-white text-sm"
      >
        🔄 Purge All Signals
      </ConfirmButton>
      <div className="space-y-3">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className="flex justify-between items-center p-3 bg-gray-800 rounded-lg shadow"
          >
            <div>
              <div className="text-white font-semibold">{agent.name}</div>
              <div className="text-xs text-gray-400">
                Status:{" "}
                <span className={agent.enabled ? "text-green-400" : "text-red-400"}>
                  {agent.enabled ? "Enabled" : "Disabled"}
                </span>
              </div>
            </div>
            <div className="space-x-2">
              <ConfirmButton
                title="Toggle Agent"
                description={`This will ${agent.enabled ? "disable" : "enable"} ${agent.name}. Proceed?`}
                onConfirm={() => onToggle(agent.id)}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-sm rounded text-white"
              >
                Toggle
              </ConfirmButton>
              <ConfirmButton
                title="Restart Agent"
                description={`Are you sure you want to restart ${agent.name}?`}
                onConfirm={() => onRestart(agent.id)}
                className="px-3 py-1 bg-yellow-500 hover:bg-yellow-600 text-sm rounded text-black"
              >
                Restart
              </ConfirmButton>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentControls;
