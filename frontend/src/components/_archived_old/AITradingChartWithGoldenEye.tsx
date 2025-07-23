/**
 * AITradingChart with Golden Eye Integration Example
 *
 * This shows how to enhance the existing AITradingChart with Golden Eye capabilities.
 * In production, these changes would be integrated directly into AITradingChart.tsx
 */

import React, { useRef, useImperativeHandle, forwardRef } from 'react';
import { AITradingChart } from './AITradingChart';
import { useGoldenEyeIntegration } from './hooks/useGoldenEyeIntegration';
import { ChartAction } from '../GoldenEyeChat/GoldenEyeChat';
import logger from '../../services/logger';


interface AITradingChartWithGoldenEyeProps {
  height?: string | number;
  symbol?: string;
  onSymbolAnalyze?: (symbol: string, analysis: any) => void;
  onGoldenEyeReady?: (controller: any) => void;
}

export interface AITradingChartHandle {
  executeGoldenEyeAction: (action: ChartAction) => Promise<void>;
  getSnapshot: () => Promise<string>;
  resetGoldenEye: () => void;
  goldenEyeController: any;
}

export const AITradingChartWithGoldenEye = forwardRef<
  AITradingChartHandle,
  AITradingChartWithGoldenEyeProps
>((props, ref) => {
  const { onGoldenEyeReady, ...chartProps } = props;

  // References to chart canvases
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const aiCanvasRef = useRef<HTMLCanvasElement>(null);
  const chartDataRef = useRef<any[]>([]);
  const currentPriceRef = useRef<number>(0);

  // Golden Eye integration
  const goldenEyeRef = useRef<any>(null);
  const { controller, isReady } = useGoldenEyeIntegration(
    {
      chartCanvasRef: mainCanvasRef,
      aiCanvasRef: aiCanvasRef,
      chartData: chartDataRef.current,
      currentPrice: currentPriceRef.current,
      onAnalyzeRequest: (params) => {
        // Handle analyze requests from chart interactions
        logger.info('Analyze request:', params);
      }
    },
    goldenEyeRef
  );

  // Notify when Golden Eye is ready
  React.useEffect(() => {
    if (isReady && controller && onGoldenEyeReady) {
      onGoldenEyeReady(controller);
    }
  }, [isReady, controller, onGoldenEyeReady]);

  // Expose methods through ref
  useImperativeHandle(ref, () => ({
    executeGoldenEyeAction: async (action: ChartAction) => {
      if (goldenEyeRef.current) {
        await goldenEyeRef.current.executeAction(action);
      }
    },
    getSnapshot: async () => {
      if (goldenEyeRef.current) {
        return goldenEyeRef.current.getSnapshot();
      }
      return '';
    },
    resetGoldenEye: () => {
      if (goldenEyeRef.current) {
        goldenEyeRef.current.reset();
      }
    },
    goldenEyeController: controller
  }), [controller]);

  // Wrapper component that provides canvas refs
  return (
    <div style={{ position: 'relative', height: props.height || '100%' }}>
      {/* Original AITradingChart */}
      <AITradingChart {...chartProps} />

      {/* Overlay canvases for Golden Eye features */}
      <canvas
        ref={mainCanvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 10
        }}
      />
      <canvas
        ref={aiCanvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 11
        }}
      />

      {/* Golden Eye Status Indicator */}
      {isReady && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            right: 10,
            padding: '4px 8px',
            background: 'rgba(255, 215, 0, 0.2)',
            border: '1px solid #FFD700',
            borderRadius: 4,
            fontSize: 12,
            color: '#FFD700',
            zIndex: 20
          }}
        >
          Golden Eye Active
        </div>
      )}
    </div>
  );
});

AITradingChartWithGoldenEye.displayName = 'AITradingChartWithGoldenEye';

/**
 * Example usage with Golden Eye Chat integration
 */
export const ExampleUsage: React.FC = () => {
  const chartRef = useRef<AITradingChartHandle>(null);

  const handleChartAction = async (action: ChartAction) => {
    if (chartRef.current) {
      await chartRef.current.executeGoldenEyeAction(action);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: 1 }}>
        <AITradingChartWithGoldenEye
          ref={chartRef}
          symbol="AAPL"
          height="100%"
          onGoldenEyeReady={(controller) => {
            logger.info('Golden Eye controller ready:', controller);

            // Enable interactive features
            controller.enableClickToAnalyze();
            controller.enableDragToSelectTimeRange();
          }}
        />
      </div>
      <div style={{ width: 400 }}>
        {/* Golden Eye Chat would go here */}
        <div style={{ padding: 20 }}>
          <h3>Golden Eye Chat</h3>
          <button
            onClick={() => handleChartAction({
              type: 'draw_prediction',
              data: {
                symbol: 'AAPL',
                prediction: [150, 152, 154, 156, 158],
                confidence_bands: {
                  upper: [151, 153, 155, 157, 159],
                  lower: [149, 151, 153, 155, 157]
                },
                horizon: 5
              }
            })}
          >
            Test Prediction
          </button>
        </div>
      </div>
    </div>
  );
};
