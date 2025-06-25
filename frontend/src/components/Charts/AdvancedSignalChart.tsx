import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, ToggleButton, ToggleButtonGroup, useTheme } from '@mui/material';
import { ShowChart, BubbleChart, Timeline, Radar } from '@mui/icons-material';
import * as d3 from 'd3';
import * as THREE from 'three';

interface SignalDataPoint {
  timestamp: Date;
  confidence: number;
  accuracy: number;
  volume: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  agents: string[];
}

interface AdvancedSignalChartProps {
  data: SignalDataPoint[];
  type: 'timeline' | '3d' | 'bubble' | 'radar';
  height?: number;
}

const AdvancedSignalChart: React.FC<AdvancedSignalChartProps> = ({
  data,
  type = 'timeline',
  height = 400,
}) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const [chartType, setChartType] = useState(type);

  useEffect(() => {
    if (!containerRef.current || !data.length) return;

    // Clear previous chart
    d3.select(containerRef.current).selectAll('*').remove();

    switch (chartType) {
      case 'timeline':
        createTimelineChart();
        break;
      case 'bubble':
        createBubbleChart();
        break;
      case 'radar':
        createRadarChart();
        break;
      case '3d':
        create3DChart();
        break;
    }
  }, [data, chartType]);

  const createTimelineChart = () => {
    const container = d3.select(containerRef.current);
    const width = container.node()!.getBoundingClientRect().width;
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.timestamp) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([innerHeight, 0]);

    // Color scale for signals
    const colorScale = d3.scaleOrdinal<string>()
      .domain(['BUY', 'SELL', 'HOLD'])
      .range(['#4CAF50', '#F44336', '#FFA500']);

    // Line generators
    const confidenceLine = d3.line<SignalDataPoint>()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.confidence))
      .curve(d3.curveMonotoneX);

    const accuracyLine = d3.line<SignalDataPoint>()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.accuracy))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')))
      .style('color', '#fff');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', '#fff');

    // Add confidence line
    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#FFD700')
      .attr('stroke-width', 2)
      .attr('d', confidenceLine);

    // Add accuracy line
    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#4CAF50')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('d', accuracyLine);

    // Add signal markers
    g.selectAll('.signal-marker')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'signal-marker')
      .attr('cx', d => xScale(d.timestamp))
      .attr('cy', d => yScale(d.confidence))
      .attr('r', 5)
      .attr('fill', d => colorScale(d.signal))
      .attr('opacity', 0.8)
      .on('mouseenter', function(event, d) {
        // Tooltip
        const tooltip = container.append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0,0,0,0.9)')
          .style('color', '#fff')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('left', `${event.offsetX + 10}px`)
          .style('top', `${event.offsetY - 10}px`)
          .html(`
            Signal: ${d.signal}<br/>
            Confidence: ${d.confidence.toFixed(1)}%<br/>
            Accuracy: ${d.accuracy.toFixed(1)}%<br/>
            Agents: ${d.agents.length}
          `);
      })
      .on('mouseleave', function() {
        container.select('.tooltip').remove();
      });

    // Add legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - 100}, 20)`);

    const legendData = [
      { label: 'Confidence', color: '#FFD700' },
      { label: 'Accuracy', color: '#4CAF50', dash: true },
    ];

    legendData.forEach((item, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendRow.append('line')
        .attr('x1', 0)
        .attr('x2', 20)
        .attr('stroke', item.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', item.dash ? '5,5' : '0');

      legendRow.append('text')
        .attr('x', 25)
        .attr('y', 5)
        .attr('fill', '#fff')
        .attr('font-size', '12px')
        .text(item.label);
    });
  };

  const createBubbleChart = () => {
    const container = d3.select(containerRef.current);
    const width = container.node()!.getBoundingClientRect().width;

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Pack layout
    const pack = d3.pack<any>()
      .size([width, height])
      .padding(2);

    // Prepare hierarchical data
    const root = d3.hierarchy({ children: data })
      .sum((d: any) => d.volume || 1)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const nodes = pack(root).leaves();

    // Color scale
    const colorScale = d3.scaleOrdinal<string>()
      .domain(['BUY', 'SELL', 'HOLD'])
      .range(['#4CAF50', '#F44336', '#FFA500']);

    // Create bubbles
    const bubbles = svg.selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x},${d.y})`);

    bubbles.append('circle')
      .attr('r', d => d.r)
      .attr('fill', d => colorScale(d.data.signal))
      .attr('opacity', 0.7)
      .attr('stroke', '#FFD700')
      .attr('stroke-width', 1);

    bubbles.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', d => Math.min(d.r / 2, 14))
      .text(d => d.data.signal);

    bubbles.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('y', d => d.r / 3)
      .attr('fill', '#fff')
      .attr('font-size', d => Math.min(d.r / 3, 10))
      .text(d => `${d.data.confidence.toFixed(0)}%`);
  };

  const createRadarChart = () => {
    const container = d3.select(containerRef.current);
    const width = Math.min(container.node()!.getBoundingClientRect().width, height);
    const margin = 50;
    const radius = (width - 2 * margin) / 2;

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Aggregate data by agent
    const agentData = d3.rollup(
      data,
      v => d3.mean(v, d => d.accuracy) || 0,
      d => d.agents[0] // Simplified for demo
    );

    const agents = Array.from(agentData.keys()).slice(0, 6); // Limit to 6 for visibility
    const angleSlice = (Math.PI * 2) / agents.length;

    // Scales
    const rScale = d3.scaleLinear()
      .domain([0, 100])
      .range([0, radius]);

    // Grid circles
    const levels = 5;
    for (let level = 1; level <= levels; level++) {
      g.append('circle')
        .attr('r', (radius / levels) * level)
        .attr('fill', 'none')
        .attr('stroke', 'rgba(255, 255, 255, 0.1)');
    }

    // Axes
    agents.forEach((agent, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;

      g.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', x)
        .attr('y2', y)
        .attr('stroke', 'rgba(255, 255, 255, 0.1)');

      g.append('text')
        .attr('x', x * 1.1)
        .attr('y', y * 1.1)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '12px')
        .text(agent);
    });

    // Data polygon
    const dataPoints = agents.map((agent, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const value = agentData.get(agent) || 0;
      return {
        x: Math.cos(angle) * rScale(value),
        y: Math.sin(angle) * rScale(value),
      };
    });

    const line = d3.line<any>()
      .x(d => d.x)
      .y(d => d.y)
      .curve(d3.curveLinearClosed);

    g.append('path')
      .datum(dataPoints)
      .attr('d', line)
      .attr('fill', '#FFD700')
      .attr('fill-opacity', 0.3)
      .attr('stroke', '#FFD700')
      .attr('stroke-width', 2);

    // Data points
    g.selectAll('.radar-point')
      .data(dataPoints)
      .enter()
      .append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', 4)
      .attr('fill', '#FFD700');
  };

  const create3DChart = () => {
    if (!containerRef.current) return;

    const width = containerRef.current.getBoundingClientRect().width;
    
    // Three.js scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);
    
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffd700, 0.8);
    pointLight.position.set(2, 3, 4);
    scene.add(pointLight);
    
    // Create 3D visualization
    const geometry = new THREE.SphereGeometry(0.1, 32, 32);
    
    data.forEach((point, index) => {
      const material = new THREE.MeshPhongMaterial({
        color: point.signal === 'BUY' ? 0x4caf50 : 
               point.signal === 'SELL' ? 0xf44336 : 0xffa500,
        emissive: 0xffd700,
        emissiveIntensity: point.confidence / 100,
      });
      
      const sphere = new THREE.Mesh(geometry, material);
      
      // Position based on time and confidence
      sphere.position.x = (index / data.length) * 6 - 3;
      sphere.position.y = (point.confidence / 100) * 3 - 1.5;
      sphere.position.z = (point.accuracy / 100) * 2 - 1;
      
      // Scale based on volume
      const scale = 0.5 + (point.volume / 1000000);
      sphere.scale.set(scale, scale, scale);
      
      scene.add(sphere);
    });
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate camera around scene
      const time = Date.now() * 0.001;
      camera.position.x = Math.cos(time * 0.5) * 5;
      camera.position.z = Math.sin(time * 0.5) * 5;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Cleanup
    return () => {
      renderer.dispose();
      containerRef.current?.removeChild(renderer.domElement);
    };
  };

  return (
    <Box>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">Signal Visualization</Typography>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(_, newType) => newType && setChartType(newType)}
          size="small"
        >
          <ToggleButton value="timeline">
            <Timeline />
          </ToggleButton>
          <ToggleButton value="bubble">
            <BubbleChart />
          </ToggleButton>
          <ToggleButton value="radar">
            <Radar />
          </ToggleButton>
          <ToggleButton value="3d">
            <ShowChart />
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box ref={containerRef} sx={{ width: '100%', height, position: 'relative' }} />
    </Box>
  );
};

export default AdvancedSignalChart;
