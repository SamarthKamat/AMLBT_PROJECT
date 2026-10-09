import React from 'react';
import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, ReferenceLine,
} from 'recharts';
import { fmt, label } from './format';

const COLORS = ['#fbbf24', '#a78bfa', '#34d399', '#f472b6', '#60a5fa'];
const GRID = { strokeDasharray: '3 3', opacity: 0.12 };
const tip = { contentStyle: { background: '#1b0b2e', border: '1px solid rgba(255,255,255,.15)', borderRadius: 10, fontSize: 12 }, formatter: (v) => (Array.isArray(v) ? v.map((x) => fmt(x)).join(' – ') : fmt(v)) };
const tick = { fontSize: 11, fill: '#bcb1d4' };

function Frame({ chart, children, height = 270 }) {
  return (
    <figure className="chart">
      {chart.title && <figcaption>{chart.title}</figcaption>}
      <ResponsiveContainer width="100%" height={height}>{children}</ResponsiveContainer>
    </figure>
  );
}

function BarView({ chart }) {
  const data = chart.data || [];
  const horizontal = chart.horizontal;
  const ys = Array.isArray(chart.y) ? chart.y : [chart.y];
  const height = horizontal ? Math.max(240, 34 * data.length + 50) : 270;
  return (
    <Frame chart={chart} height={height}>
      <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ left: horizontal ? 20 : 0, right: 12 }}>
        <CartesianGrid {...GRID} />
        {horizontal ? <XAxis type="number" tick={tick} /> : <XAxis dataKey={chart.x} tick={tick} interval={0} angle={data.length > 6 ? -25 : 0} textAnchor={data.length > 6 ? 'end' : 'middle'} height={data.length > 6 ? 60 : 30} />}
        {horizontal ? <YAxis type="category" dataKey={chart.x} width={120} tick={tick} /> : <YAxis tick={tick} />}
        <Tooltip {...tip} />
        {ys.length > 1 && <Legend />}
        {ys.map((y, i) => <Bar isAnimationActive={false} key={y} dataKey={y} fill={COLORS[(i + 1) % COLORS.length]} radius={horizontal ? [0, 6, 6, 0] : [6, 6, 0, 0]} />)}
      </BarChart>
    </Frame>
  );
}

function LineView({ chart }) {
  const ys = Array.isArray(chart.y) ? chart.y : [chart.y];
  return (
    <Frame chart={chart}>
      <LineChart data={chart.data || []}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey={chart.x} tick={tick} minTickGap={28} />
        <YAxis tick={tick} domain={['auto', 'auto']} />
        <Tooltip {...tip} />
        {ys.length > 1 && <Legend />}
        {ys.map((y, i) => <Line isAnimationActive={false} key={y} type="monotone" dataKey={y} stroke={COLORS[i % COLORS.length]} strokeWidth={i === 0 ? 2.5 : 2} strokeDasharray={i > 0 ? '6 4' : undefined} dot={false} connectNulls />)}
      </LineChart>
    </Frame>
  );
}

function ScatterView({ chart }) {
  const data = chart.data || [];
  const hl = chart.highlight;
  const normal = hl ? data.filter((d) => !d[hl]) : data;
  const flagged = hl ? data.filter((d) => d[hl]) : [];
  const nums = data.flatMap((d) => [d[chart.x], d[chart.y]]).filter((v) => typeof v === 'number');
  const lo = Math.min(...nums), hi = Math.max(...nums);
  const lbl = chart.label;
  const Tip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const p = payload[0].payload;
    return (
      <div className="chartTip">
        {lbl && p[lbl] !== undefined && <b>{String(p[lbl])}</b>}
        <div>{label(chart.x)}: {fmt(p[chart.x])}</div>
        <div>{label(chart.y)}: {fmt(p[chart.y])}</div>
        {hl && <div>{p[hl] ? 'Flagged as unusual' : 'Within the usual range'}</div>}
      </div>
    );
  };
  return (
    <Frame chart={chart}>
      <ScatterChart margin={{ left: 0, right: 12 }}>
        <CartesianGrid {...GRID} />
        <XAxis type="number" dataKey={chart.x} name={chart.x} tick={tick} domain={chart.identity ? [lo, hi] : ['auto', 'auto']} label={{ value: label(chart.x), position: 'insideBottom', offset: -2, fill: '#9f93b2', fontSize: 11 }} height={44} />
        <YAxis type="number" dataKey={chart.y} name={chart.y} tick={tick} domain={chart.identity ? [lo, hi] : ['auto', 'auto']} />
        <ZAxis range={[28, 28]} />
        <Tooltip content={<Tip />} />
        {chart.identity && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#34d399" strokeDasharray="5 4" />}
        <Scatter isAnimationActive={false} data={normal} fill="#a78bfa" fillOpacity={0.65} />
        {flagged.length > 0 && <Scatter isAnimationActive={false} data={flagged} fill="#f87171" shape="diamond" />}
      </ScatterChart>
    </Frame>
  );
}

function ForecastView({ chart }) {
  const data = (chart.data || []).map((d) => ({ ...d, bandBase: d.lower != null && d.upper != null ? d.lower : null, bandSpan: d.lower != null && d.upper != null ? d.upper - d.lower : null }));
  return (
    <Frame chart={chart} height={300}>
      <ComposedChart data={data}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey={chart.x || 'period'} tick={tick} minTickGap={28} />
        <YAxis tick={tick} domain={['auto', 'auto']} />
        <Tooltip {...tip} />
        <Legend />
        <Area dataKey="bandBase" stackId="band" stroke="none" fill="transparent" legendType="none" tooltipType="none" isAnimationActive={false} />
        <Area dataKey="bandSpan" stackId="band" name="80% interval" stroke="none" fill="#fbbf24" fillOpacity={0.2} legendType="rect" tooltipType="none" isAnimationActive={false} />
        <Line isAnimationActive={false} type="monotone" dataKey="actual" name="Actual" stroke="#a78bfa" strokeWidth={2.4} dot={false} connectNulls />
        <Line isAnimationActive={false} type="monotone" dataKey="forecast" name="Forecast" stroke="#fbbf24" strokeWidth={2.8} dot={{ r: 3 }} connectNulls />
      </ComposedChart>
    </Frame>
  );
}

export default function Chart({ chart }) {
  if (!chart?.data?.length) return null;
  switch (chart.type) {
    case 'bar': return <BarView chart={chart} />;
    case 'line': return <LineView chart={chart} />;
    case 'scatter': return <ScatterView chart={chart} />;
    case 'forecast': return <ForecastView chart={chart} />;
    default: return null;
  }
}
