# Golden Signals AI - Frontend Documentation

## 🏗 Architecture Overview

The frontend is built with:
- **React 18.3** with TypeScript
- **Material-UI v5** with custom golden theme
- **Redux Toolkit** for state management
- **React Query** for server state
- **WebSockets** for real-time updates
- **D3.js & Three.js** for visualizations

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/         # Reusable components
│   │   ├── AI/            # AI-specific components
│   │   ├── Agents/        # Agent visualizations
│   │   ├── Charts/        # Data visualizations
│   │   ├── Common/        # Shared components
│   │   ├── DesignSystem/  # UI component library
│   │   ├── Layout/        # Layout components
│   │   ├── Notifications/ # Notification system
│   │   └── Signals/       # Signal components
│   ├── contexts/          # React contexts
│   ├── hooks/             # Custom React hooks
│   ├── pages/             # Page components
│   ├── services/          # API services
│   ├── store/             # Redux store
│   ├── theme/             # Theme configuration
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions
├── public/                # Static assets
└── cypress/               # E2E tests
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation
```bash
cd frontend
npm install
```

### Development
```bash
npm start
# App runs on http://localhost:3000
```

### Build
```bash
npm run build
# Production build in ./build
```

### Testing
```bash
# Unit tests
npm test

# E2E tests
npm run cypress:open
```

## 🎨 Component Library

### Core Components

#### GoldenButton
Premium button with multiple variants:
```tsx
import { GoldenButton } from '@/components/DesignSystem/GoldenComponents';

<GoldenButton variant="gradient" onClick={handleClick}>
  Click Me
</GoldenButton>

// Variants: contained, outlined, text, gradient, glow
```

#### GoldenCard
Card with glassmorphism effect:
```tsx
import { GoldenCard } from '@/components/DesignSystem/GoldenComponents';

<GoldenCard>
  <Typography>Content</Typography>
</GoldenCard>
```

#### SignalCard
Display signal information:
```tsx
import SignalCard from '@/components/Signals/SignalCard';

<SignalCard 
  signal={signalData}
  onAction={handleAction}
  compact={false}
/>
```

## 🔌 API Integration

### REST API
```typescript
import { api } from '@/services/api';

// Get signals
const signals = await api.signals.getAll();

// Get specific signal
const signal = await api.signals.getById(id);
```

### WebSocket
```typescript
import { useWebSocket } from '@/hooks/useWebSocket';

const { data, isConnected } = useWebSocket('signals.all', {
  onMessage: (data) => {
    console.log('New signal:', data);
  }
});
```

## 🎯 State Management

### Redux Store Structure
```typescript
{
  signals: {
    items: Signal[],
    loading: boolean,
    error: string | null
  },
  agents: {
    consensus: ConsensusData,
    status: AgentStatus[]
  },
  user: {
    profile: UserProfile,
    preferences: UserPreferences
  }
}
```

### Using Redux
```typescript
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { signalActions } from '@/store/slices/signalSlice';

// Read state
const signals = useAppSelector(state => state.signals.items);

// Dispatch action
const dispatch = useAppDispatch();
dispatch(signalActions.addSignal(newSignal));
```

## 🎨 Theming

### Using the Golden Theme
```typescript
import { useTheme } from '@mui/material/styles';

const theme = useTheme();

// Access theme values
theme.palette.primary.main // #FFD700
theme.palette.background.default // #0D1117
```

### Custom Styling
```typescript
import { styled } from '@mui/material/styles';
import { utilityClasses } from '@/theme/goldenTheme';

const StyledComponent = styled('div')(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  padding: theme.spacing(2),
}));
```

## 📊 Data Visualization

### D3.js Charts
```typescript
import { AdvancedSignalChart } from '@/components/Charts/AdvancedSignalChart';

<AdvancedSignalChart
  data={chartData}
  type="timeline" // timeline, bubble, radar, 3d
  height={400}
/>
```

### Real-time Updates
Charts automatically update when new data arrives via WebSocket.

## 🧪 Testing

### Unit Testing
```typescript
import { render, screen } from '@testing-library/react';
import { SignalCard } from '@/components/Signals/SignalCard';

test('renders signal card', () => {
  render(<SignalCard signal={mockSignal} />);
  expect(screen.getByText(mockSignal.symbol)).toBeInTheDocument();
});
```

### E2E Testing
```typescript
describe('Signal Flow', () => {
  it('displays new signals in real-time', () => {
    cy.visit('/signals');
    cy.get('[data-testid="signal-card"]').should('have.length.gte', 1);
  });
});
```

## 🚀 Performance

### Code Splitting
Pages are automatically code-split:
```typescript
const SignalStream = lazy(() => import('./pages/SignalStream'));
```

### Optimization Tips
1. Use `React.memo` for expensive components
2. Implement virtual scrolling for long lists
3. Debounce search inputs
4. Lazy load images and heavy components
5. Use WebWorkers for computations

## 🔐 Security

- All API calls use HTTPS
- Authentication tokens stored securely
- XSS protection via React's built-in escaping
- CSRF tokens for state-changing operations

## 📱 Responsive Design

Breakpoints:
- **xs**: 0px
- **sm**: 600px
- **md**: 960px
- **lg**: 1280px
- **xl**: 1920px

## 🚢 Deployment

### Environment Variables
```env
REACT_APP_API_URL=https://api.goldensignals.ai
REACT_APP_WS_URL=wss://ws.goldensignals.ai
REACT_APP_ENABLE_ANALYTICS=true
```

### Build Optimization
```bash
npm run build
# Generates optimized production build
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

### Code Style
- Use TypeScript strict mode
- Follow ESLint rules
- Format with Prettier
- Write tests for new features

## 📚 Resources

- [React Documentation](https://react.dev)
- [Material-UI Documentation](https://mui.com)
- [Redux Toolkit](https://redux-toolkit.js.org)
- [D3.js Documentation](https://d3js.org)

## 🆘 Troubleshooting

### Common Issues

1. **WebSocket connection fails**
   - Check WS_URL environment variable
   - Verify backend is running

2. **Build errors**
   - Clear node_modules and reinstall
   - Check TypeScript errors

3. **Performance issues**
   - Enable React DevTools Profiler
   - Check for unnecessary re-renders

## 📞 Support

For questions or issues:
- Check existing GitHub issues
- Create new issue with reproduction steps
- Contact development team
