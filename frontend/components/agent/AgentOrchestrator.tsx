import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

/**
 * AgentContext provides a registry and communication layer for all UI agents.
 */
const AgentContext = createContext<any>(null);

export function useAgentRegistry() {
  return useContext(AgentContext);
}

interface AgentOrchestratorProps {
  children: ReactNode;
}

/**
 * AgentOrchestrator manages registration and coordination of UI agents (components)
 * and provides a dark theme context.
 */
export function AgentOrchestrator({ children }: AgentOrchestratorProps) {
  const [agents, setAgents] = useState<{ [key: string]: any }>({});
  const [darkMode, setDarkMode] = useState(true);

  // Register an agent by name
  const registerAgent = (name: string, agentApi: any) => {
    setAgents(prev => ({ ...prev, [name]: agentApi }));
  };

  // Allow agents to communicate via the registry
  const sendMessage = (to: string, msg: any) => {
    if (agents[to] && agents[to].onMessage) {
      agents[to].onMessage(msg);
    }
  };

  return (
    <AgentContext.Provider value={{ agents, registerAgent, sendMessage, darkMode, setDarkMode }}>
      <div className={darkMode ? 'dark bg-zinc-900 text-white min-h-screen' : 'bg-zinc-50 text-zinc-900 min-h-screen'}>
        {children}
      </div>
    </AgentContext.Provider>
  );
}
