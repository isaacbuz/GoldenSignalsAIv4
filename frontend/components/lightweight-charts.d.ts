// Minimal type declarations for lightweight-charts to resolve TypeScript errors

declare module 'lightweight-charts' {
  export function createChart(container: HTMLElement, options?: any): any;
  // Add more type definitions as needed for your usage
}
