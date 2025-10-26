import React, { useEffect, useRef } from 'react';
import { Card, Spin, Empty } from 'antd';
import * as d3 from 'd3';

/**
 * Yield Curve Chart Component
 * Interactive D3.js chart for visualizing yield curves
 */
const YieldCurveChart = ({ data, loading = false, title = "Yield Curve" }) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0 || loading) return;

    drawChart();
  }, [data, loading]);

  const drawChart = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous chart

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create main group
    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Prepare data
    const curveData = data.map(d => ({
      tenor: parseFloat(d.tenor),
      rate: parseFloat(d.rate) * 100 // Convert to percentage
    })).sort((a, b) => a.tenor - b.tenor);

    // Scales
    const xScale = d3.scaleLinear()
      .domain(d3.extent(curveData, d => d.tenor))
      .range([0, width])
      .nice();

    const yScale = d3.scaleLinear()
      .domain(d3.extent(curveData, d => d.rate))
      .range([height, 0])
      .nice();

    // Line generator
    const line = d3.line()
      .x(d => xScale(d.tenor))
      .y(d => yScale(d.rate))
      .curve(d3.curveMonotoneX);

    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.1)
      .call(d3.axisLeft(yScale)
        .tickSize(-width)
        .tickFormat("")
      );

    g.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.1)
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-height)
        .tickFormat("")
      );

    // Add X axis
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .append("text")
      .attr("x", width / 2)
      .attr("y", 35)
      .attr("fill", "#666")
      .attr("text-anchor", "middle")
      .text("Tenor (Years)");

    // Add Y axis
    g.append("g")
      .call(d3.axisLeft(yScale))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -35)
      .attr("fill", "#666")
      .attr("text-anchor", "middle")
      .text("Yield (%)");

    // Add the line
    g.append("path")
      .datum(curveData)
      .attr("fill", "none")
      .attr("stroke", "#1890ff")
      .attr("stroke-width", 3)
      .attr("d", line);

    // Add data points
    g.selectAll(".dot")
      .data(curveData)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", d => xScale(d.tenor))
      .attr("cy", d => yScale(d.rate))
      .attr("r", 5)
      .attr("fill", "#1890ff")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        // Highlight point
        d3.select(this)
          .transition()
          .duration(200)
          .attr("r", 8)
          .attr("fill", "#40a9ff");

        // Show tooltip
        showTooltip(event, d);
      })
      .on("mouseout", function() {
        // Reset point
        d3.select(this)
          .transition()
          .duration(200)
          .attr("r", 5)
          .attr("fill", "#1890ff");

        // Hide tooltip
        hideTooltip();
      });

    // Add area under curve
    const area = d3.area()
      .x(d => xScale(d.tenor))
      .y0(height)
      .y1(d => yScale(d.rate))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(curveData)
      .attr("fill", "rgba(24, 144, 255, 0.1)")
      .attr("d", area);
  };

  const showTooltip = (event, d) => {
    const tooltip = d3.select(tooltipRef.current);
    tooltip
      .style("opacity", 1)
      .html(`
        <strong>Tenor:</strong> ${d.tenor} years<br/>
        <strong>Yield:</strong> ${d.rate.toFixed(2)}%
      `)
      .style("left", (event.pageX + 10) + "px")
      .style("top", (event.pageY - 10) + "px");
  };

  const hideTooltip = () => {
    d3.select(tooltipRef.current)
      .style("opacity", 0);
  };

  if (loading) {
    return (
      <Card title={title} style={{ height: 450 }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 350 }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card title={title} style={{ height: 450 }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 350 }}>
          <Empty description="No yield curve data available" />
        </div>
      </Card>
    );
  }

  return (
    <Card title={title} style={{ height: 450 }}>
      <div style={{ position: 'relative' }}>
        <svg ref={svgRef} style={{ width: '100%', height: 350 }}></svg>
        <div
          ref={tooltipRef}
          style={{
            position: 'absolute',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            opacity: 0,
            zIndex: 1000
          }}
        />
      </div>
      <div style={{ textAlign: 'center', marginTop: 10, fontSize: '12px', color: '#666' }}>
        Hover over points to see detailed values
      </div>
    </Card>
  );
};

export default YieldCurveChart;