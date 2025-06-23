# GoldenSignalsAI Frontend Migration - LLM Implementation Prompt

## Context for Implementation

You are being tasked with implementing a comprehensive frontend migration for GoldenSignalsAI, a sophisticated AI-powered trading platform. This document provides all the context and specifications you need to understand the project and execute the migration successfully.

## Project Understanding

### What is GoldenSignalsAI?

GoldenSignalsAI is a next-generation financial technology platform that combines artificial intelligence with real-time market analysis to provide:

1. **AI-Powered Trading Signals**
   - Multiple AI agents analyze market conditions
   - Consensus-based signal generation
   - Real-time buy/sell recommendations
   - Confidence scoring for each signal

2. **Advanced Market Analytics**
   - Professional-grade charting tools
   - Technical indicators and overlays
   - Volume analysis and market depth
   - Pattern recognition and forecasting

3. **Portfolio Management**
   - Real-time position tracking
   - P&L calculations and performance metrics
   - Risk assessment and management
   - Historical performance analysis

4. **Options Trading Support**
   - Options chain analysis
   - Greeks calculations
   - Volatility surface visualization
   - Options strategy recommendations

### Current Technical State

The existing frontend has the following characteristics:

**Technology Stack:**
- React with JavaScript (limited TypeScript)
- Vite as build tool
- Multiple charting libraries (Recharts, lightweight-charts, MUI x-charts)
- Basic API integration
- Minimal state management
- No comprehensive testing

**File Structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── AI/
│   │   ├── Chart/
│   │   ├── Common/
│   │   ├── HybridDashboard/
│   │   ├── Layout/
│   │   ├── Main/
│   │   ├── Options/
│   │   └── SignalsChart/
│   ├── contexts/
│   ├── pages/
│   ├── services/
│   └── store/
```

**Key Problems:**
1. Inconsistent UI/UX patterns
2. Multiple charting libraries causing maintenance issues
3. No unified state management
4. Poor TypeScript coverage
5. Lack of testing
6. Performance issues
7. Limited error handling
8. No accessibility considerations

## Your Implementation Task

You need to migrate this frontend to a modern, scalable architecture following these specifications:

### Target Architecture

**Core Technologies:**
```yaml
Framework: React 18+ with TypeScript (strict mode)
Build Tool: Vite 5+
State Management:
  - Global State: Zustand
  - Server State: TanStack Query (React Query) v5
UI Framework: Material-UI (MUI) v5
Charting: TradingView Lightweight Charts (single library)
WebSocket: Native WebSocket with reconnection logic
Testing: Jest + React Testing Library + Cypress
Code Quality: ESLint + Prettier + Husky
```

### New Folder Structure

```
frontend/
├── src/
│   ├── app/                    # Application setup
│   │   ├── App.tsx            # Root component
│   │   ├── Router.tsx         # Route configuration
│   │   └── Providers.tsx      # Context providers
│   ├── components/            # Reusable components
│   │   ├── atoms/            # Basic components
│   │   ├── molecules/        # Composite components
│   │   └── organisms/        # Complex components
│   ├── features/             # Feature modules
│   │   ├── dashboard/
│   │   ├── signals/
│   │   ├── analytics/
│   │   └── portfolio/
│   ├── services/             # External services
│   │   ├── api/
│   │   └── websocket/
│   ├── store/               # State management
│   ├── hooks/               # Custom hooks
│   ├── utils/               # Utilities
│   ├── types/               # TypeScript types
│   └── theme/               # Theme configuration
```

### Implementation Priorities

1. **Phase 1: Foundation (Critical)**
   - Set up TypeScript configuration with strict mode
   - Create the new folder structure
   - Implement the API client with proper typing
   - Set up Zustand store
   - Configure React Query
   - Create base theme configuration

2. **Phase 2: Core Components**
   - Build atomic design system components
   - Create unified TradingView chart component
   - Implement error boundaries
   - Add loading states and skeletons

3. **Phase 3: Feature Migration**
   - Migrate Dashboard with real-time updates
   - Rebuild Signals module with filtering/sorting
   - Implement Analytics with new charting
   - Create Portfolio management views

4. **Phase 4: Enhancement**
   - Add comprehensive error handling
   - Implement accessibility features
   - Add performance optimizations
   - Create test suites

### Key Implementation Details

#### 1. API Client Pattern
```typescript
// All API calls should follow this pattern
import { apiClient } from '@/services/api/client';
import { useQuery, useMutation } from '@tanstack/react-query';

// Query hook example
export const useMarketData = (symbol: string) => {
  return useQuery({
    queryKey: ['market', symbol],
    queryFn: () => apiClient.get(`/market/${symbol}`).then(res => res.data),
    staleTime: 5000,
  });
};

// Mutation hook example
export const useCreateOrder = () => {
  return useMutation({
    mutationFn: (order: OrderRequest) => 
      apiClient.post('/orders', order).then(res => res.data),
  });
};
```

#### 2. State Management Pattern
```typescript
// Zustand store slices
interface AppState {
  // User slice
  user: User | null;
  setUser: (user: User | null) => void;
  
  // Market slice
  selectedSymbol: string;
  watchlist: string[];
  setSelectedSymbol: (symbol: string) => void;
  addToWatchlist: (symbol: string) => void;
  
  // UI slice
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
  toggleTheme: () => void;
  toggleSidebar: () => void;
}
```

#### 3. Component Pattern
```typescript
// All components should follow this structure
interface ComponentProps {
  // Strongly typed props
}

export const Component: React.FC<ComponentProps> = memo(({
  // Destructured props
}) => {
  // Hooks at the top
  // Logic in the middle
  // Return JSX
  
  return (
    <StyledContainer>
      {/* Component content */}
    </StyledContainer>
  );
});

Component.displayName = 'Component';
```

#### 4. WebSocket Integration
```typescript
// Real-time updates pattern
const useWebSocket = (channel: string) => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.channel === channel) {
        setData(message.data);
      }
    };
    
    return () => ws.close();
  }, [channel]);
  
  return data;
};
```

### Critical Requirements

1. **TypeScript Everywhere**
   - No `any` types allowed
   - Strict mode enabled
   - All props and returns typed

2. **Performance First**
   - Use React.memo for expensive components
   - Implement virtual scrolling for lists
   - Lazy load routes and heavy components
   - Optimize bundle size < 500KB

3. **Accessibility**
   - ARIA labels on all interactive elements
   - Keyboard navigation support
   - Screen reader compatibility
   - Color contrast compliance

4. **Error Handling**
   - Global error boundary
   - Graceful degradation
   - User-friendly error messages
   - Retry mechanisms

5. **Testing**
   - Unit tests for utilities
   - Component tests for UI
   - Integration tests for features
   - E2E tests for critical paths

### UI/UX Guidelines

1. **Design System**
   - Consistent spacing (4px base unit)
   - Unified color palette
   - Typography scale
   - Component variants

2. **Responsive Design**
   - Mobile-first approach
   - Breakpoints: 640px, 768px, 1024px, 1280px
   - Fluid typography and spacing

3. **Interactions**
   - Loading states for all async operations
   - Optimistic updates where appropriate
   - Smooth transitions and animations
   - Clear feedback for user actions

### Backend API Endpoints

The frontend will communicate with these main endpoints:

```
GET    /api/v1/market/{symbol}          # Market data
GET    /api/v1/signals                  # AI signals list
GET    /api/v1/signals/{id}            # Signal details
GET    /api/v1/portfolio                # User portfolio
POST   /api/v1/orders                   # Create order
GET    /api/v1/analytics/{symbol}       # Analytics data
WS     /ws/market                       # Real-time market updates
WS     /ws/signals                      # Real-time signals
```

### Migration Execution Steps

1. **Do NOT delete the existing code** until the new implementation is complete
2. Create the new structure alongside the old one
3. Migrate features incrementally with feature flags
4. Ensure backward compatibility during transition
5. Run old and new versions in parallel for testing
6. Only remove old code after thorough testing

### Expected Deliverables

1. **Week 1-2: Foundation**
   - New project structure
   - Core services (API, WebSocket)
   - State management setup
   - Theme configuration

2. **Week 3-4: Components**
   - Atomic design system
   - Unified chart component
   - Layout components
   - Common UI elements

3. **Week 5-8: Features**
   - Dashboard module
   - Signals module
   - Analytics module
   - Portfolio module

4. **Week 9-10: Polish**
   - Testing suite
   - Performance optimization
   - Documentation
   - Deployment setup

### Success Criteria

- **Performance**: Initial load < 2s, TTI < 3s
- **Type Safety**: 100% TypeScript coverage
- **Testing**: >80% code coverage
- **Accessibility**: WCAG 2.1 AA compliant
- **Bundle Size**: <500KB gzipped
- **Error Rate**: <0.1% of sessions

## Additional Context

- The platform handles real-time financial data, so performance and reliability are critical
- Users expect professional-grade charting similar to TradingView
- The AI signals are the core differentiator - they must be prominently displayed
- Mobile responsiveness is important but desktop is the primary use case
- Dark mode is essential for traders who work long hours

## Your Response Should Include

1. Confirmation that you understand the project scope
2. Any clarifying questions about requirements
3. A proposed implementation approach
4. Code examples for key components
5. Potential challenges and solutions

Remember: This is a professional financial platform where users make real money decisions. Quality, reliability, and user trust are paramount. 