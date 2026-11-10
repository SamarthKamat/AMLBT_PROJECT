import React, { useMemo, useState } from 'react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, LineChart, Pie, PieChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import { fmt, label } from './format';

export const COLORS = ['#fbbf24', '#a78bfa', '#34d399', '#f472b6', '#60a5fa', '#fb923c', '#2dd4bf', '#e879f9', '#facc15', '#93c5fd'];
const GRID = { strokeDasharray: '3 3', opacity: 0.12 };
const tipStyle = { background: '#1b0b2e', border: '1px solid rgba(255,255,255,.15)', borderRadius: 10, fontSize: 12, color: '#f3edff' };
const tip = { contentStyle: tipStyle, itemStyle: { color: '#f3edff' }, labelStyle: { color: '#cfc5e0' }, formatter: (v) => (Array.isArray(v) ? v.map((x) => fmt(x)).join(' – ') : fmt(v)) };
const tick = { fontSize: 11, fill: '#bcb1d4' };
const ys = (chart) => (Array.isArray(chart.y) ? chart.y : chart.y ? [chart.y] : []);

function Frame({ chart, children, height = 280, tools }) {
  return (
    <figure className="chart">
      <div className="chartHead">{chart.title && <figcaption>{chart.title}</figcaption>}{tools}</div>
      <ResponsiveContainer width="100%" height={height}>{children}</ResponsiveContainer>
      {chart.note && <div className="tableNote">{chart.note}</div>}
    </figure>
  );
}

function BarView({ chart, tools }) {
  const data = chart.data || [];
  const horizontal = chart.horizontal;
  const keys = ys(chart);
  const height = horizontal ? Math.max(240, 30 * data.length + 60) : 290;
  const neg = data.some((d) => keys.some((k) => d[k] < 0));
  return (
    <Frame chart={chart} height={height} tools={tools}>
      <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ left: horizontal ? 24 : 4, right: 16, top: 6 }}>
        <CartesianGrid {...GRID} />
        {horizontal ? <XAxis type="number" tick={tick} tickFormatter={(v) => fmt(v)} /> : <XAxis dataKey={chart.x} tick={tick} interval={0} angle={data.length > 6 ? -28 : 0} textAnchor={data.length > 6 ? 'end' : 'middle'} height={data.length > 6 ? 70 : 30} />}
        {horizontal ? <YAxis type="category" dataKey={chart.x} width={140} tick={tick} /> : <YAxis tick={tick} tickFormatter={(v) => fmt(v)} />}
        <Tooltip {...tip} cursor={{ fill: 'rgba(255,255,255,.04)' }} />
        {keys.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {neg && <ReferenceLine {...(horizontal ? { x: 0 } : { y: 0 })} stroke="#9f93b2" />}
        {keys.map((y, i) => (
          <Bar isAnimationActive={false} key={y} dataKey={y} stackId={chart.stacked ? 's' : undefined} fill={COLORS[(i + (keys.length > 1 ? 0 : 1)) % COLORS.length]}
            radius={chart.stacked ? 0 : horizontal ? [0, 6, 6, 0] : [6, 6, 0, 0]}>
            {keys.length === 1 && data.map((d, j) => <Cell key={j} fill={d[y] < 0 ? '#f87171' : COLORS[1]} />)}
          </Bar>
        ))}
      </BarChart>
    </Frame>
  );
}

function LineView({ chart, tools, area }) {
  const keys = ys(chart);
  const Chart = area ? AreaChart : LineChart;
  return (
    <Frame chart={chart} tools={tools}>
      <Chart data={chart.data || []} margin={{ right: 16, top: 6 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey={chart.x} tick={tick} minTickGap={24} />
        <YAxis tick={tick} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v)} />
        <Tooltip {...tip} />
        {keys.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {keys.map((y, i) => (area
          ? <Area isAnimationActive={false} key={y} type="monotone" dataKey={y} stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]} fillOpacity={0.18} connectNulls />
          : <Line isAnimationActive={false} key={y} type="monotone" dataKey={y} stroke={COLORS[i % COLORS.length]} strokeWidth={i === 0 ? 2.5 : 2} strokeDasharray={i > 0 && keys.length === 2 ? '6 4' : undefined} dot={false} connectNulls />))}
      </Chart>
    </Frame>
  );
}

function PieView({ chart, tools }) {
  const y = ys(chart)[0];
  const data = (chart.data || []).filter((d) => typeof d[y] === 'number' && d[y] > 0).slice(0, 10);
  const total = data.reduce((a, d) => a + d[y], 0) || 1;
  return (
    <Frame chart={chart} height={300} tools={tools}>
      <PieChart>
        <Pie isAnimationActive={false} data={data} dataKey={y} nameKey={chart.x} innerRadius="45%" outerRadius="78%" paddingAngle={2}
          label={({ payload, value }) => `${String(payload[chart.x]).slice(0, 16)} ${(100 * value / total).toFixed(0)}%`} labelLine={false} fontSize={11}>
          {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="rgba(0,0,0,.25)" />)}
        </Pie>
        <Tooltip {...tip} />
      </PieChart>
    </Frame>
  );
}

function ScatterView({ chart, tools }) {
  const data = chart.data || [];
  const hl = chart.highlight;
  const grp = chart.group;
  const nums = data.flatMap((d) => [d[chart.x], d[chart.y]]).filter((v) => typeof v === 'number');
  const lo = Math.min(...nums), hi = Math.max(...nums);
  const groups = grp ? [...new Set(data.map((d) => d[grp]))] : null;
  const Tip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const p = payload[0].payload;
    return (
      <div className="chartTip">
        {chart.label && p[chart.label] !== undefined && <b>{String(p[chart.label])}</b>}
        {grp && <b>{String(p[grp])}</b>}
        <div>{label(chart.x)}: {fmt(p[chart.x])}</div>
        <div>{label(chart.yLabel || chart.y)}: {fmt(p[chart.y])}</div>
        {hl && <div>{p[hl] ? 'Flagged as unusual' : 'Within the usual range'}</div>}
      </div>
    );
  };
  return (
    <Frame chart={chart} height={300} tools={tools}>
      <ScatterChart margin={{ left: 4, right: 16, top: 6 }}>
        <CartesianGrid {...GRID} />
        <XAxis type="number" dataKey={chart.x} name={chart.x} tick={tick} domain={chart.identity ? [lo, hi] : ['auto', 'auto']} tickFormatter={(v) => fmt(v)}
          label={{ value: label(chart.x), position: 'insideBottom', offset: -2, fill: '#9f93b2', fontSize: 11 }} height={44} />
        <YAxis type="number" dataKey={chart.y} name={chart.y} tick={tick} domain={chart.identity ? [lo, hi] : ['auto', 'auto']} tickFormatter={(v) => fmt(v)} />
        <ZAxis range={[26, 26]} />
        <Tooltip content={<Tip />} />
        {chart.identity && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#34d399" strokeDasharray="5 4" />}
        {groups ? groups.map((gname, i) => <Scatter key={gname} name={gname} isAnimationActive={false} data={data.filter((d) => d[grp] === gname)} fill={COLORS[i % COLORS.length]} fillOpacity={0.75} />)
          : <Scatter isAnimationActive={false} data={hl ? data.filter((d) => !d[hl]) : data} fill="#a78bfa" fillOpacity={0.65} />}
        {hl && <Scatter isAnimationActive={false} data={data.filter((d) => d[hl])} fill="#f87171" shape="diamond" />}
        {groups && <Legend wrapperStyle={{ fontSize: 11 }} />}
      </ScatterChart>
    </Frame>
  );
}

function ForecastView({ chart, tools }) {
  const data = (chart.data || []).map((d) => ({
    ...d,
    b80: d.lower != null && d.upper != null ? [d.lower, d.upper] : null,
    b95: d.lower95 != null && d.upper95 != null ? [d.lower95, d.upper95] : null,
  }));
  return (
    <Frame chart={chart} height={310} tools={tools}>
      <ComposedChart data={data} margin={{ right: 16, top: 6 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey={chart.x || 'period'} tick={tick} minTickGap={26} />
        <YAxis tick={tick} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v)} />
        <Tooltip {...tip} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Area dataKey="b95" name="95% range" stroke="none" fill="#fbbf24" fillOpacity={0.1} isAnimationActive={false} connectNulls />
        <Area dataKey="b80" name="80% range" stroke="none" fill="#fbbf24" fillOpacity={0.22} isAnimationActive={false} connectNulls />
        <Line isAnimationActive={false} type="monotone" dataKey="actual" name="Actual" stroke="#a78bfa" strokeWidth={2.4} dot={false} connectNulls />
        <Line isAnimationActive={false} type="monotone" dataKey="forecast" name="Forecast" stroke="#fbbf24" strokeWidth={2.8} dot={{ r: 3 }} connectNulls />
      </ComposedChart>
    </Frame>
  );
}

function heatColor(v, absolute, min, max) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'rgba(255,255,255,.03)';
  if (absolute) {
    const t = max > min ? (v - min) / (max - min) : 0.5;
    return `rgba(251,191,36,${0.08 + 0.82 * t})`;
  }
  const t = Math.max(-1, Math.min(1, v));
  return t >= 0 ? `rgba(52,211,153,${0.08 + 0.85 * t})` : `rgba(248,113,113,${0.08 + 0.85 * -t})`;
}

function HeatmapView({ chart, tools }) {
  const cols = chart.columns || [];
  const rows = chart.rows || cols;
  const flat = (chart.matrix || []).flat().filter((v) => typeof v === 'number');
  const min = Math.min(...flat), max = Math.max(...flat);
  return (
    <figure className="chart">
      <div className="chartHead"><figcaption>{chart.title}</figcaption>{tools}</div>
      <div className="heatWrap">
        <table className="heat">
          <thead><tr><th />{cols.map((c) => <th key={c} title={c}><span>{String(c).slice(0, 14)}</span></th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r}>
                <th title={r}>{String(r).slice(0, 18)}</th>
                {cols.map((c, j) => {
                  const v = chart.matrix?.[i]?.[j];
                  return <td key={c} style={{ background: heatColor(v, chart.absolute, min, max) }} title={`${r} × ${c}: ${fmt(v, 3)}`}>{v === null || v === undefined ? '' : chart.absolute ? fmt(v, 0) : fmt(v, 2)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {chart.note && <div className="tableNote">{chart.note}</div>}
    </figure>
  );
}

function BoxView({ chart, tools }) {
  const items = chart.data || [];
  if (!items.length) return null;
  const W = 640, rowH = 34, pad = 150, H = items.length * rowH + 34;
  const scaleFor = (b) => {
    const lo = chart.independent ? b.min : Math.min(...items.map((x) => x.min));
    const hi = chart.independent ? b.max : Math.max(...items.map((x) => x.max));
    return (v) => pad + ((v - lo) / ((hi - lo) || 1)) * (W - pad - 20);
  };
  return (
    <figure className="chart">
      <div className="chartHead"><figcaption>{chart.title}</figcaption>{tools}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="boxSvg" role="img" aria-label={chart.title}>
        {items.map((b, i) => {
          const s = scaleFor(b);
          const y = 14 + i * rowH;
          return (
            <g key={b.name}>
              <text x={pad - 10} y={y + 14} textAnchor="end" className="boxLabel">{String(b.name).slice(0, 22)}</text>
              <line x1={s(b.min)} x2={s(b.max)} y1={y + 10} y2={y + 10} stroke="#9f93b2" />
              <line x1={s(b.min)} x2={s(b.min)} y1={y + 4} y2={y + 16} stroke="#9f93b2" />
              <line x1={s(b.max)} x2={s(b.max)} y1={y + 4} y2={y + 16} stroke="#9f93b2" />
              <rect x={s(b.q1)} y={y} width={Math.max(2, s(b.q3) - s(b.q1))} height={20} rx={4} fill="rgba(167,139,250,.35)" stroke="#a78bfa" />
              <line x1={s(b.median)} x2={s(b.median)} y1={y} y2={y + 20} stroke="#fbbf24" strokeWidth={2.5} />
              <title>{`${b.name}: min ${fmt(b.min)}, Q1 ${fmt(b.q1)}, median ${fmt(b.median)}, Q3 ${fmt(b.q3)}, max ${fmt(b.max)}, ${b.outliers} outlier(s)`}</title>
              <text x={W - 6} y={y + 14} textAnchor="end" className="boxNote">{b.outliers ? `${b.outliers} out` : ''}</text>
            </g>
          );
        })}
      </svg>
      <div className="tableNote">{chart.independent ? 'Each row has its own scale (whiskers = 1.5×IQR). Hover a row for exact values.' : 'Whiskers = 1.5×IQR; hover for exact values.'}</div>
    </figure>
  );
}

function WaterfallView({ chart, tools }) {
  const data = chart.data || [];
  const color = (k) => (k === 'up' ? '#34d399' : k === 'down' ? '#f87171' : '#a78bfa');
  return (
    <Frame chart={chart} height={Math.max(280, 26 * data.length + 80)} tools={tools}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24, top: 6 }}>
        <CartesianGrid {...GRID} />
        <XAxis type="number" tick={tick} tickFormatter={(v) => fmt(v)} domain={['auto', 'auto']} />
        <YAxis type="category" dataKey={chart.x || 'label'} width={190} tick={tick} />
        <Tooltip {...tip} formatter={(v, n, p) => [fmt(p.payload.value), p.payload.kind === 'base' || p.payload.kind === 'total' ? 'value' : 'contribution']} />
        <Bar dataKey="start" stackId="w" fill="transparent" isAnimationActive={false} tooltipType="none" />
        <Bar dataKey="delta" stackId="w" isAnimationActive={false} radius={[0, 4, 4, 0]}>
          {data.map((d, i) => <Cell key={i} fill={color(d.kind)} />)}
        </Bar>
      </BarChart>
    </Frame>
  );
}

function TableView({ chart, tools }) {
  const rows = chart.data || [];
  const keys = rows.length ? Object.keys(rows[0]).slice(0, 8) : [];
  return (
    <figure className="chart">
      <div className="chartHead"><figcaption>{chart.title}</figcaption>{tools}</div>
      <div className="tableWrap"><table><thead><tr>{keys.map((k) => <th key={k}>{label(k)}</th>)}</tr></thead>
        <tbody>{rows.slice(0, 30).map((r, i) => <tr key={i}>{keys.map((k) => <td key={k}>{fmt(r[k], 3)}</td>)}</tr>)}</tbody></table></div>
    </figure>
  );
}

// Which alternative renderings make sense for a chart (the user can switch with the chips).
function alternatives(chart) {
  const n = (chart.data || []).length;
  const k = ys(chart).length;
  if (chart.type === 'bar' && !chart.stacked) {
    const alts = ['bar'];
    if (k === 1 && n <= 10 && (chart.data || []).every((d) => (d[ys(chart)[0]] ?? 0) >= 0)) alts.push('pie');
    alts.push('line', 'table');
    return alts;
  }
  if (chart.type === 'bar') return ['bar', 'table'];
  if (chart.type === 'line') return ['line', 'area', 'bar', 'table'];
  if (chart.type === 'pie') return ['pie', 'bar', 'table'];
  return [chart.type];
}

export default function Chart({ chart, preferred }) {
  const alts = useMemo(() => (chart ? alternatives(chart) : []), [chart]);
  const initial = preferred && alts.includes(preferred) ? preferred : (preferred === 'histogram' && chart?.type === 'bar' ? 'bar' : chart?.type);
  const [view, setView] = useState(initial);
  if (!chart) return null;
  const hasData = chart.type === 'heatmap' ? chart.matrix?.length : chart.data?.length;
  if (!hasData) return null;
  const tools = alts.length > 1 ? (
    <div className="viewSwitch">{alts.map((a) => <button key={a} className={view === a ? 'on' : ''} onClick={() => setView(a)}>{a}</button>)}</div>
  ) : null;
  const c = { ...chart };
  if (view === 'pie' && chart.type !== 'pie') return <PieView chart={c} tools={tools} />;
  if (view === 'bar' && chart.type !== 'bar') return <BarView chart={{ ...c, horizontal: (c.data || []).length > 8 }} tools={tools} />;
  if (view === 'line' && chart.type !== 'line') return <LineView chart={c} tools={tools} />;
  if (view === 'area') return <LineView chart={c} tools={tools} area />;
  if (view === 'table') return <TableView chart={c} tools={tools} />;
  switch (chart.type) {
    case 'bar': return <BarView chart={c} tools={tools} />;
    case 'line': return <LineView chart={c} tools={tools} />;
    case 'area': return <LineView chart={c} tools={tools} area />;
    case 'pie': return <PieView chart={c} tools={tools} />;
    case 'scatter': return <ScatterView chart={c} tools={tools} />;
    case 'forecast': return <ForecastView chart={c} tools={tools} />;
    case 'heatmap': return <HeatmapView chart={c} tools={tools} />;
    case 'box': return <BoxView chart={c} tools={tools} />;
    case 'waterfall': return <WaterfallView chart={c} tools={tools} />;
    default: return null;
  }
}
