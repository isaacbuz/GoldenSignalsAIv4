# Golden Eye Chart Integration Game Plan

## Overview
This plan outlines how to enhance the AITradingChart component to support all Golden Eye AI Prophet Chat capabilities, including real-time agent visualizations, predictive overlays, and interactive chart actions.

## Current State Analysis

### Existing Capabilities
1. **Canvas Architecture**: Dual-canvas system (main + AI overlay)
2. **Agent Integration**: Basic agent signal display
3. **Drawing Hooks**: Modular drawing system with specialized hooks
4. **Real-time Updates**: WebSocket integration for live data
5. **Technical Indicators**: RSI, MACD, Volume, etc.

### Missing Capabilities for Golden Eye
1. **Prediction Visualization**: Multi-point prediction lines with confidence bands
2. **Pattern Highlighting**: Area selection for detected patterns
3. **Dynamic Level Drawing**: Support/resistance from agent analysis
4. **Multi-Symbol Comparison**: Overlay multiple symbols with agent insights
5. **Live Annotation**: Real-time text/shape annotations from chat
6. **Chart Actions API**: Programmatic control from Golden Eye

## Implementation Phases

### Phase 1: Chart Action API (2-3 days)

#### 1.1 Create Chart Controller Interface
```typescript
// frontend/src/components/AIChart/controllers/ChartController.ts
export interface IChartController {
  // Prediction methods
  drawPrediction(params: PredictionParams): void;
  clearPrediction(): void;

  // Signal methods
  addAgentSignal(signal: AgentSignal): void;
  highlightSignals(agentName: string): void;

  // Level methods
  drawSupportResistance(levels: PriceLevel[]): void;
  drawEntryExitZones(zones: TradingZone[]): void;

  // Pattern methods
  highlightPattern(pattern: ChartPattern): void;
  clearPatterns(): void;

  // Annotation methods
  addAnnotation(annotation: ChartAnnotation): void;

  // View control
  zoomToTimeRange(start: Date, end: Date): void;
  panToPrice(price: number): void;
}
```

#### 1.2 Implement Chart Actions
- [ ] Create ChartController class
- [ ] Add ref forwarding to expose controller
- [ ] Implement action queue for smooth animations
- [ ] Add undo/redo capability

#### 1.3 Golden Eye Integration Points
```typescript
// In AITradingChart component
useImperativeHandle(ref, () => ({
  controller: chartController,
  getSnapshot: () => canvas.toDataURL(),
  getAnalysis: () => currentAnalysis,
}));
```

### Phase 2: Enhanced Drawing Capabilities (3-4 days)

#### 2.1 Prediction Drawing System
```typescript
// frontend/src/components/AIChart/components/ChartCanvas/hooks/usePredictionDrawing.ts
export const usePredictionDrawing = () => {
  const drawPrediction = useCallback((
    ctx: CanvasRenderingContext2D,
    prediction: {
      points: PricePoint[];
      confidence: {
        upper: PricePoint[];
        lower: PricePoint[];
      };
      style: PredictionStyle;
    }
  ) => {
    // Draw confidence bands
    drawConfidenceBands(ctx, prediction.confidence);

    // Draw main prediction line with gradient
    drawPredictionLine(ctx, prediction.points);

    // Add confidence percentage labels
    drawConfidenceLabels(ctx, prediction);

    // Animate the drawing
    animatePrediction(ctx, prediction);
  }, []);

  return { drawPrediction };
};
```

#### 2.2 Pattern Highlighting
- [ ] Implement pattern detection overlay
- [ ] Add morphing animations for pattern boundaries
- [ ] Create pattern legend component
- [ ] Support multiple simultaneous patterns

#### 2.3 Advanced Level Drawing
- [ ] Dynamic level calculations from agent data
- [ ] Level strength visualization (opacity/thickness)
- [ ] Interactive level tooltips
- [ ] Level breach alerts

### Phase 3: Multi-Agent Visualization (3-4 days)

#### 3.1 Agent Consensus Display
```typescript
// frontend/src/components/AIChart/components/AgentConsensus/AgentConsensusPanel.tsx
export const AgentConsensusPanel: React.FC<{
  consensus: AgentConsensus;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}> = ({ consensus, position }) => {
  return (
    <FloatingPanel position={position}>
      <ConsensusGauge value={consensus.score} />
      <AgentVotes agents={consensus.votes} />
      <ConfidenceIndicator level={consensus.confidence} />
    </FloatingPanel>
  );
};
```

#### 3.2 Agent Signal Clustering
- [ ] Group nearby signals from same agent
- [ ] Show signal strength with visual weight
- [ ] Implement signal filtering by agent
- [ ] Add agent performance overlay

#### 3.3 Real-time Agent Updates
- [ ] WebSocket channel for agent signals
- [ ] Signal arrival animations
- [ ] Agent "thinking" indicators
- [ ] Historical signal replay

### Phase 4: Interactive Features (2-3 days)

#### 4.1 Click-to-Analyze
```typescript
// Add to chart mouse handlers
const handleChartClick = (event: MouseEvent) => {
  const { price, time } = getChartCoordinates(event);

  // Trigger Golden Eye analysis at clicked point
  goldenEyeChat.analyzePoint({
    symbol: currentSymbol,
    price,
    time,
    context: 'user_clicked_chart'
  });
};
```

#### 4.2 Drag-to-Select Analysis
- [ ] Rectangle selection tool
- [ ] Time range analysis trigger
- [ ] Pattern search in selection
- [ ] Export selected data

#### 4.3 Live Annotation from Chat
- [ ] Text annotation tools
- [ ] Shape drawing (arrows, circles)
- [ ] Annotation persistence
- [ ] Collaborative annotations

### Phase 5: Advanced Overlays (3-4 days)

#### 5.1 AI Confidence Heatmap
```typescript
// frontend/src/components/AIChart/overlays/ConfidenceHeatmap.tsx
export const ConfidenceHeatmap: React.FC<{
  data: ConfidenceData[];
  opacity: number;
}> = ({ data, opacity }) => {
  // Render confidence levels as color gradient overlay
  // Higher confidence = more intense color
  // Show which areas AI is most certain about
};
```

#### 5.2 Multi-Symbol Comparison
- [ ] Symbol overlay system
- [ ] Normalized price scales
- [ ] Correlation indicators
- [ ] Relative performance metrics

#### 5.3 Prediction Accuracy Tracking
- [ ] Show historical prediction accuracy
- [ ] Confidence calibration display
- [ ] Agent performance comparison
- [ ] Accuracy trend lines

### Phase 6: Performance & Polish (2 days)

#### 6.1 Rendering Optimization
- [ ] Implement dirty rectangle rendering
- [ ] Add level-of-detail system
- [ ] Optimize for 60fps with all features
- [ ] Memory usage profiling

#### 6.2 Mobile Responsiveness
- [ ] Touch gesture support
- [ ] Responsive overlay positioning
- [ ] Mobile-optimized controls
- [ ] Performance mode for mobile

#### 6.3 Accessibility
- [ ] Keyboard navigation
- [ ] Screen reader support
- [ ] High contrast mode
- [ ] Configurable visual indicators

## Technical Architecture

### Component Structure
```
AIChart/
├── AITradingChart.tsx              # Main component
├── controllers/
│   ├── ChartController.ts          # Chart action API
│   └── GoldenEyeController.ts      # Golden Eye specific
├── hooks/
│   ├── useGoldenEyeActions.ts      # Golden Eye integration
│   ├── usePredictionDrawing.ts     # Prediction rendering
│   └── usePatternHighlight.ts      # Pattern visualization
├── overlays/
│   ├── PredictionOverlay.tsx       # Prediction display
│   ├── AgentConsensusOverlay.tsx   # Agent consensus
│   └── ConfidenceHeatmap.tsx       # AI confidence
└── utils/
    ├── chartActions.ts             # Action implementations
    └── coordinateTransform.ts      # Coordinate utilities
```

### State Management
```typescript
// Enhanced chart context
interface ChartState {
  // Existing state
  data: ChartDataPoint[];
  indicators: string[];

  // Golden Eye additions
  predictions: PredictionData[];
  agentSignals: AgentSignal[];
  patterns: ChartPattern[];
  annotations: ChartAnnotation[];
  activeAgents: string[];
  consensusData: ConsensusData;
}
```

### Event System
```typescript
// Chart event emitter
class ChartEventEmitter extends EventEmitter {
  // User interactions
  emit('click', { price, time, candle });
  emit('select', { startTime, endTime, priceRange });

  // Golden Eye events
  emit('predictionDrawn', { prediction });
  emit('signalAdded', { signal });
  emit('patternDetected', { pattern });

  // Analysis triggers
  emit('analyzeRequest', { type, data });
}
```

## Integration Timeline

### Week 1
- Day 1-2: Chart Controller API
- Day 3-4: Prediction drawing system
- Day 5: Pattern highlighting

### Week 2
- Day 1-2: Multi-agent visualization
- Day 3-4: Interactive features
- Day 5: Testing & debugging

### Week 3
- Day 1-2: Advanced overlays
- Day 3-4: Performance optimization
- Day 5: Documentation & examples

## Success Metrics

1. **Performance**
   - 60fps with all features active
   - <100ms response to Golden Eye commands
   - <1s to render complex predictions

2. **Functionality**
   - All Golden Eye chart actions supported
   - Smooth animations for all transitions
   - No visual glitches or artifacts

3. **User Experience**
   - Intuitive interaction model
   - Clear visual hierarchy
   - Responsive to all inputs

## Testing Strategy

### Unit Tests
- Test each drawing hook independently
- Verify coordinate transformations
- Test action queue processing

### Integration Tests
- Golden Eye command execution
- Multi-agent signal display
- Pattern detection accuracy

### Visual Regression Tests
- Screenshot comparisons
- Animation smoothness
- Cross-browser rendering

## Migration Guide

### For Existing Chart Users
```typescript
// Old way
<AITradingChart symbol="AAPL" />

// New way with Golden Eye
const chartRef = useRef();

<AITradingChart
  ref={chartRef}
  symbol="AAPL"
  onReady={() => {
    // Chart controller now available
    chartRef.current.controller.setGoldenEyeMode(true);
  }}
/>

<GoldenEyeChat
  onChartAction={(action) => {
    chartRef.current.controller.executeAction(action);
  }}
/>
```

## Risk Mitigation

1. **Performance Degradation**
   - Solution: Implement feature flags for gradual rollout
   - Add performance mode toggle

2. **Breaking Changes**
   - Solution: Maintain backward compatibility
   - Version the API changes

3. **Complexity Overload**
   - Solution: Progressive disclosure of features
   - Smart defaults for common use cases

## Conclusion

This plan transforms the AITradingChart into a fully integrated Golden Eye-compatible visualization system. The phased approach ensures we can deliver value incrementally while maintaining stability and performance.
