import * as React from "react";
import { cn } from "../../lib/utils";

interface TabProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

export function Tabs({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("flex border-b border-fintech-border", className)}>{children}</div>;
}

export function Tab({ label, active, onClick }: TabProps) {
  return (
    <button
      className={cn(
        "px-4 py-2 font-medium transition-colors",
        active ? "border-b-2 border-fintech-accent text-fintech-accent" : "text-white hover:text-fintech-accent"
      )}
      onClick={onClick}
    >
      {label}
    </button>
  );
}
