import React, { useEffect, useState } from 'react';
import { BarChart3 } from 'lucide-react';
import Chart from '../Charts';
import { DataTable, Metrics } from '../ResultView';
import { getJson } from '../api';
import { Loading, ErrorBox } from './common';

export default function EdaTab({ session, version, columns }) {
  const [col, setCol] = useState('');
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    setRes(null); setErr(null);
    getJson(`/api/session/${session}/eda${col ? `?column=${encodeURIComponent(col)}` : ''}`).then(setRes).catch(setErr);
  }, [session, version, col]);
  return (
    <div className="tabBody">
      <div className="tabIntro"><BarChart3 size={18} /><div><h3>Exploratory data analysis</h3>
        <p>Distributions, skewness, outliers, category mixes, missingness, associations and the main KPI over time — or pick a column for a focused deep dive.</p></div>
        <select className="select" value={col} onChange={(e) => setCol(e.target.value)}>
          <option value="">Whole dataset</option>{columns.map((c) => <option key={c} value={c}>{c}</option>)}
        </select></div>
      {err && <ErrorBox error={err} />}
      {!res && !err && <Loading text="Running exploratory analysis…" />}
      {res && (
        <>
          <p className="lead">{res.summary}</p>
          <Metrics metrics={res.metrics} />
          {res.insights?.length > 0 && <div className="insights"><div className="blockTitle">Auto-generated insights</div><ul>{res.insights.map((t, i) => <li key={i}>{t}</li>)}</ul></div>}
          <div className="chartGrid">{res.charts.map((c, i) => <Chart key={i} chart={c} />)}</div>
          <DataTable rows={res.table} title={res.table_title} wide max={25} csvName="eda_numeric" />
          {Object.entries(res.tables || {}).map(([k, rows]) => <DataTable key={k} rows={rows} title={k.replace(/_/g, ' ')} max={20} />)}
        </>
      )}
    </div>
  );
}
