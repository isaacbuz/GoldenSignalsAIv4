import * as React from "react";
import { cn } from "../../lib/utils";

export function Panel({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("bg-fintech-panel rounded-lg shadow border border-fintech-border p-6", className)}>
      {children}
    </div>
  );
}
