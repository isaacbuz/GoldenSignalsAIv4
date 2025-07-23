import React, { useEffect, useRef } from 'react';
import { Box, Typography, alpha } from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';

const drawPattern = keyframes`
  0% {
    stroke-dashoffset: 1000;
    opacity: 0;
  }
  50% {
    opacity: 1;
  }
  100% {
    stroke-dashoffset: 0;
    opacity: 1;
  }
`;

const pulsePattern = keyframes`
  0% {
    opacity: 0.8;
  }
  50% {
    opacity: 1;
    filter: drop-shadow(0 0 10px currentColor);
  }
  100% {
    opacity: 0.8;
  }
`;

const PatternSvg = styled('svg')({
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  pointerEvents: 'none',
  zIndex: 10,
});

const PatternLine = styled('path')<{ delay?: number }>(({ theme, delay = 0 }) => ({
  fill: 'none',
  stroke: theme.palette.warning.main,
  strokeWidth: 2,
  strokeDasharray: 1000,
  strokeDashoffset: 1000,
  filter: `drop-shadow(0 0 3px ${alpha(theme.palette.warning.main, 0.5)})`,
  animation: `${drawPattern} 2s ease-out ${delay}s forwards, ${pulsePattern} 2s ease-in-out ${delay + 2}s infinite`,
}));

const PatternLabel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  padding: theme.spacing(0.5, 1),
  backgroundColor: alpha(theme.palette.warning.main, 0.9),
  color: theme.palette.common.white,
  borderRadius: theme.spacing(0.5),
  fontSize: '0.75rem',
  fontWeight: 'bold',
  pointerEvents: 'none',
  zIndex: 11,
  animation: `${drawPattern} 1s ease-out 2s forwards`,
  opacity: 0,
}));

interface PatternOverlayProps {
  pattern: {
    type: string;
    points: { x: number; y: number }[];
    label?: string;
    confidence: number;
  };
  containerWidth: number;
  containerHeight: number;
}

export const PatternOverlay: React.FC<PatternOverlayProps> = ({
  pattern,
  containerWidth,
  containerHeight,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!pattern.points || pattern.points.length < 2) return;

    // Generate SVG path
    const pathData = pattern.points
      .map((point, index) => {
        const x = point.x * containerWidth;
        const y = point.y * containerHeight;
        return index === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
      })
      .join(' ');

    // Set path data
    const pathElement = svgRef.current?.querySelector('path');
    if (pathElement) {
      pathElement.setAttribute('d', pathData);
    }
  }, [pattern, containerWidth, containerHeight]);

  const getLabelPosition = () => {
    if (!pattern.points || pattern.points.length === 0) return { left: 0, top: 0 };

    // Position label at the center of the pattern
    const avgX = pattern.points.reduce((sum, p) => sum + p.x, 0) / pattern.points.length;
    const avgY = pattern.points.reduce((sum, p) => sum + p.y, 0) / pattern.points.length;

    return {
      left: avgX * containerWidth,
      top: avgY * containerHeight,
    };
  };

  const getPatternPaths = () => {
    switch (pattern.type) {
      case 'triangle':
        // For triangles, draw both trendlines
        if (pattern.points.length >= 4) {
          const upperTrend = `M ${pattern.points[0].x * containerWidth} ${pattern.points[0].y * containerHeight} L ${pattern.points[1].x * containerWidth} ${pattern.points[1].y * containerHeight}`;
          const lowerTrend = `M ${pattern.points[2].x * containerWidth} ${pattern.points[2].y * containerHeight} L ${pattern.points[3].x * containerWidth} ${pattern.points[3].y * containerHeight}`;
          return [
            <PatternLine key="upper" d={upperTrend} delay={0} />,
            <PatternLine key="lower" d={lowerTrend} delay={0.5} />,
          ];
        }
        break;
      case 'head_shoulders':
        // Draw neckline and shoulders
        if (pattern.points.length >= 5) {
          const paths = [];
          // Connect all points
          for (let i = 0; i < pattern.points.length - 1; i++) {
            const path = `M ${pattern.points[i].x * containerWidth} ${pattern.points[i].y * containerHeight} L ${pattern.points[i + 1].x * containerWidth} ${pattern.points[i + 1].y * containerHeight}`;
            paths.push(<PatternLine key={`segment-${i}`} d={path} delay={i * 0.2} />);
          }
          return paths;
        }
        break;
      default:
        // Default pattern drawing
        const pathData = pattern.points
          .map((point, index) => {
            const x = point.x * containerWidth;
            const y = point.y * containerHeight;
            return index === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
          })
          .join(' ');
        return [<PatternLine key="default" d={pathData} />];
    }
    return [];
  };

  const labelPos = getLabelPosition();

  return (
    <>
      <PatternSvg ref={svgRef}>
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {getPatternPaths()}
      </PatternSvg>
      {pattern.label && (
        <PatternLabel
          style={{
            left: labelPos.left,
            top: labelPos.top,
            transform: 'translate(-50%, -50%)',
          }}
        >
          <Typography variant="caption">
            {pattern.label} ({(pattern.confidence * 100).toFixed(0)}%)
          </Typography>
        </PatternLabel>
      )}
    </>
  );
};
