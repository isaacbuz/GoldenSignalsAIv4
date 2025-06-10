# GoldenSignalsAI V3 Frontend

A modern, real-time trading dashboard built with React, TypeScript, and Material-UI.

## 🚀 Features

### Core Functionality
- **Real-time Dashboard** with live market data and trading signals
- **WebSocket Integration** for instant data updates
- **Multi-Agent AI System** monitoring and performance tracking
- **Portfolio Management** with P&L tracking and position analysis
- **Advanced Signal Filtering** with confidence scoring and risk assessment
- **Responsive Design** optimized for desktop and mobile devices

### Technical Highlights
- **TypeScript** for type safety and better developer experience
- **Material-UI (MUI)** for consistent, professional design
- **React Query** for efficient data fetching and caching
- **Socket.IO** for real-time WebSocket communication
- **React Router** for client-side navigation
- **Vite** for fast development and optimized builds

## 📁 Project Structure

```
frontend/src/
├── components/          # Reusable UI components
│   ├── Common/         # Generic components (Loading, Error Boundary)
│   └── Layout/         # Layout components (Navigation, Header)
├── pages/              # Main application pages
│   ├── Dashboard/      # Real-time trading dashboard
│   ├── Signals/        # Trading signals management
│   ├── Agents/         # AI agents monitoring
│   ├── Portfolio/      # Portfolio tracking
│   ├── Analytics/      # Performance analytics
│   └── Settings/       # Application settings
├── services/           # API and WebSocket services
│   ├── api.ts         # REST API client
│   └── websocket.ts   # WebSocket service
└── App.tsx            # Main application component
```

## 🛠️ Development Setup

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API running on http://localhost:8000

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open in browser:**
   ```
   http://localhost:3000
   ```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript compiler check

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_BASE_URL=ws://localhost:8000
```

### API Integration

The frontend connects to the FastAPI backend through:
- **REST API** for data fetching and mutations
- **WebSocket** for real-time updates
- **Automatic reconnection** with exponential backoff

## 📱 Core Pages

### 1. Dashboard (`/dashboard`)
- Real-time market overview with key metrics
- Live trading signals feed
- Agent performance summary
- Quick action buttons

### 2. Trading Signals (`/signals`)
- Comprehensive signal listing with filtering
- Card and table view modes
- Confidence scoring and risk assessment
- Real-time signal updates

### 3. AI Agents (`/agents`)
- Agent performance monitoring
- Individual agent statistics
- Orchestrator status overview
- Agent feature breakdown

### 4. Portfolio (`/portfolio`)
- Portfolio value and P&L tracking
- Current positions table
- Recent trades history
- Performance analytics

### 5. Analytics (`/analytics`)
- Advanced performance metrics (placeholder)
- Risk analysis and reporting
- Market insights and trends

### 6. Settings (`/settings`)
- User preferences and configuration (placeholder)
- Trading parameters
- Notification settings

## 🎨 Design System

### Theme
- **Dark Mode** optimized for trading environments
- **Green/Red** color scheme for buy/sell signals
- **Professional Typography** with Inter font family
- **Consistent Spacing** using 8px grid system

### Components
- **Responsive Grid System** for layout
- **Cards** for content organization
- **Data Tables** for financial data
- **Charts** for market visualization (ready for integration)
- **Real-time Indicators** for connection status

## 🔄 Real-time Features

### WebSocket Integration
- Automatic connection management
- Subscription-based data updates
- Custom React hooks for real-time data
- Connection status monitoring

### Supported Events
- `market_data` - Live price updates
- `signal` - New trading signals
- `agent_status` - Agent performance updates
- `portfolio_update` - Portfolio changes
- `alert` - System notifications

## 📊 State Management

### React Query
- Server state management
- Automatic background refetching
- Optimistic updates
- Error handling and retries

### Local State
- React hooks for component state
- Custom hooks for WebSocket subscriptions
- Context for global UI state

## 🚀 Performance Optimizations

### Code Splitting
- Lazy loading of route components
- Dynamic imports for heavy libraries
- Bundle splitting for vendor code

### Caching Strategy
- React Query for server state caching
- 5-minute stale time for market data
- Background refetching for real-time data

### WebSocket Optimization
- Connection pooling
- Automatic reconnection
- Heartbeat monitoring
- Efficient event handling

## 🔐 Security Features

### API Security
- Automatic token management
- Request/response interceptors
- Error handling for auth failures
- HTTPS enforcement in production

### Data Validation
- TypeScript interfaces for type safety
- Runtime validation for API responses
- Sanitization of user inputs

## 📱 Responsive Design

### Breakpoints
- **Mobile:** < 600px
- **Tablet:** 600px - 960px
- **Desktop:** > 960px

### Adaptive Features
- Collapsible navigation drawer
- Responsive grid layouts
- Touch-optimized controls
- Optimized table layouts

## 🧪 Testing Strategy

### Unit Testing
- Component testing with React Testing Library
- Service layer testing
- Custom hook testing

### Integration Testing
- API integration tests
- WebSocket connection tests
- End-to-end user flows

## 🚀 Deployment

### Production Build
```bash
npm run build
```

### Build Optimization
- Tree shaking for minimal bundle size
- Asset optimization and compression
- Source maps for debugging
- Progressive web app features

### Environment Configuration
- Environment-specific API endpoints
- Feature flags for staging
- Performance monitoring integration

## 🔧 Development Tools

### Developer Experience
- Hot module replacement
- TypeScript error checking
- ESLint code quality
- Prettier code formatting

### Debugging
- React DevTools integration
- Redux DevTools for state inspection
- Network request monitoring
- Real-time data flow visualization

## 📈 Performance Monitoring

### Metrics Tracked
- Bundle size and load times
- WebSocket connection health
- API response times
- User interaction analytics

### Optimization Targets
- < 3s initial load time
- < 100ms WebSocket latency
- 95%+ cache hit rate
- Zero memory leaks

## 🤝 Contributing

### Code Standards
- TypeScript strict mode
- ESLint + Prettier configuration
- Conventional commit messages
- Component documentation

### Pull Request Process
1. Feature branch from main
2. TypeScript compilation check
3. Lint and format code
4. Component testing
5. Code review approval

---

## 📞 Support

For technical support or questions about the frontend implementation, please refer to the main project documentation or contact the development team.

**Built with ❤️ for professional traders and developers** 