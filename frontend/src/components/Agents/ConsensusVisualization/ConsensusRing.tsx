import React, { useEffect, useRef } from 'react';
import { Box, Typography, Chip } from '@mui/material';
import { styled } from '@mui/material/styles';
import * as d3 from 'd3';

const VisualizationContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  minHeight: 400,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
}));

interface AgentVote {
  agentId: string;
  agentType: string;
  vote: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  color: string;
}

interface ConsensusRingProps {
  agents: AgentVote[];
  finalConsensus: {
    signal: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
  };
  animated?: boolean;
}

const ConsensusRing: React.FC<ConsensusRingProps> = ({
  agents,
  finalConsensus,
  animated = true,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 400;
    const height = 400;
    const centerX = width / 2;
    const centerY = height / 2;
    const outerRadius = 150;
    const innerRadius = 80;

    // Create main group
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${centerX}, ${centerY})`);

    // Create agent segments
    const anglePerAgent = (2 * Math.PI) / agents.length;

    agents.forEach((agent, index) => {
      const startAngle = index * anglePerAgent;
      const endAngle = (index + 1) * anglePerAgent;

      // Create arc
      const arc = d3
        .arc()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius * (0.8 + agent.confidence * 0.2))
        .startAngle(startAngle)
        .endAngle(endAngle);

      // Draw agent segment
      const segment = g
        .append('path')
        .attr('d', arc as any)
        .attr('fill', agent.color)
        .attr('opacity', 0)
        .attr('stroke', '#FFD700')
        .attr('stroke-width', 2);

      if (animated) {
        segment
          .transition()
          .duration(500)
          .delay(index * 100)
          .attr('opacity', 0.3 + agent.confidence * 0.5);
      } else {
        segment.attr('opacity', 0.3 + agent.confidence * 0.5);
      }

      // Add agent label
      const labelAngle = startAngle + anglePerAgent / 2;
      const labelRadius = outerRadius + 30;
      const labelX = Math.sin(labelAngle) * labelRadius;
      const labelY = -Math.cos(labelAngle) * labelRadius;

      g.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#FFFFFF')
        .attr('font-size', '12px')
        .attr('opacity', 0)
        .text(agent.agentType)
        .transition()
        .duration(500)
        .delay(index * 100 + 300)
        .attr('opacity', 1);

      // Add confidence indicator
      const confRadius = innerRadius - 20;
      const confX = Math.sin(labelAngle) * confRadius;
      const confY = -Math.cos(labelAngle) * confRadius;

      g.append('circle')
        .attr('cx', confX)
        .attr('cy', confY)
        .attr('r', 0)
        .attr('fill', agent.color)
        .transition()
        .duration(500)
        .delay(index * 100 + 500)
        .attr('r', 5 + agent.confidence * 10);
    });

    // Central consensus display
    const centerGroup = g.append('g');

    // Center circle
    centerGroup
      .append('circle')
      .attr('r', 0)
      .attr('fill', '#0A0E27')
      .attr('stroke', '#FFD700')
      .attr('stroke-width', 3)
      .transition()
      .duration(800)
      .delay(agents.length * 100)
      .attr('r', innerRadius - 10);

    // Consensus text
    centerGroup
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#FFD700')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('opacity', 0)
      .text(finalConsensus.signal)
      .transition()
      .duration(500)
      .delay(agents.length * 100 + 800)
      .attr('opacity', 1);

    // Confidence percentage
    centerGroup
      .append('text')
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#FFFFFF')
      .attr('font-size', '18px')
      .attr('opacity', 0)
      .text(`${finalConsensus.confidence}%`)
      .transition()
      .duration(500)
      .delay(agents.length * 100 + 1000)
      .attr('opacity', 1);

    // Animated rotation
    if (animated) {
      g.transition()
        .duration(20000)
        .ease(d3.easeLinear)
        .attrTween('transform', () => {
          return (t: number) => `translate(${centerX}, ${centerY}) rotate(${t * 360})`;
        })
        .on('end', function repeat() {
          d3.select(this)
            .transition()
            .duration(20000)
            .ease(d3.easeLinear)
            .attrTween('transform', () => {
              return (t: number) => `translate(${centerX}, ${centerY}) rotate(${t * 360})`;
            })
            .on('end', repeat);
        });
    }
  }, [agents, finalConsensus, animated]);

  return (
    <VisualizationContainer>
      <svg ref={svgRef} />
      <Box sx={{ mt: 3, display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
        <Chip label="BUY" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)', color: '#4CAF50' }} />
        <Chip label="HOLD" sx={{ backgroundColor: 'rgba(255, 165, 0, 0.1)', color: '#FFA500' }} />
        <Chip label="SELL" sx={{ backgroundColor: 'rgba(244, 67, 54, 0.1)', color: '#F44336' }} />
      </Box>
    </VisualizationContainer>
  );
};

export default ConsensusRing;
