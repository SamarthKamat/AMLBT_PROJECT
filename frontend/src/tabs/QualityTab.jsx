import React, { useEffect, useState } from 'react';
import { ShieldCheck, Wand2, RotateCcw, Download } from 'lucide-react';
import { DataTable } from '../ResultView';
import Chart from '../Charts';
import { download, getJson, postJson } from '../api';
import { fmt } from '../format';
import { Loading, ErrorBox } from './common';

const ACTION_LABEL = {
  standardize_categories: 'Standardise label spellings (North / north / NORTH)', drop_duplicates: 'Remove exact duplicate rows',
  invalid_to_missing: 'Treat invalid values as missing (negative ages, rates over 100%)', impute_missing: 'Impute missing values (median / mean / mode / "Unknown")',
  drop_sparse_columns: 'Drop columns with more than 60% missing', cap_outliers: 'Cap extreme outliers at the 1st / 99th percentile',
  drop_constant_columns: 'Drop constant columns', drop_missing_rows: 'Drop every row with a missing value',
};

function Gauge({ score, grade }) {
  const angle = (score / 100) * 180;
  const color = score >= 90 ? '#34d399' : score >= 75 ? '#fbbf24' : '#f87171';
  const r = 70, cx = 90, cy = 85;
  const rad = (a) => (Math.PI * (180 - a)) / 180;
  const x = cx + r * Math.cos(rad(angle)), y = cy - r * Math.sin(rad(angle));
  return (
    <svg viewBox="0 0 180 100" className="gauge" role="img" aria-label={`Quality score ${score.toFixed(1)}`}>
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} stroke="rgba(255,255,255,.1)" strokeWidth="14" fill="none" />
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${x} ${y}`} stroke={color} strokeWidth="14" fill="none" strokeLinecap="round" />
      <text x={cx} y={cy - 12} textAnchor="middle" className="gaugeNum">{score.toFixed(1)}</text>
      <text x={cx} y={cy + 6} textAnchor="middle" className="gaugeGrade">grade {grade}</text>
    </svg>
  );
}

export default function QualityTab({ session, version, onChanged }) {
  const [q, setQ] = useState(null);
  const [err, setErr] = useState(null);
  const [sel, setSel] = useState([]);
  const [busy, setBusy] = useState(false);
  const [cmp, setCmp] = useState(null);
  useEffect(() => {
    setQ(null); setErr(null);
    getJson(`/api/session/${session}/quality`).then((d) => { setQ(d); setSel(d.actions.default); }).catch(setErr);
  }, [session, version]);
  const toggle = (a) => setSel((s) => (s.includes(a) ? s.filter((x) => x !== a) : [...s, a]));
  const clean = async () => {
    setBusy(true); setErr(null);
    try { const r = await postJson(`/api/session/${session}/clean`, { actions: sel }); setCmp(r); onChanged?.(r); } catch (e) { setErr(e); } finally { setBusy(false); }
  };
  const revert = async () => {
    setBusy(true);
    try { const r = await postJson(`/api/session/${session}/revert`, {}); setCmp(null); onChanged?.(r); } catch (e) { setErr(e); } finally { setBusy(false); }
  };
  if (err && !q) return <ErrorBox error={err} />;
  if (!q) return <Loading text="Scoring data quality…" />;
  const dims = Object.entries(q.dimensions).map(([k, v]) => ({ dimension: k, score: v, weight: q.weights[k] }));
  return (
    <div className="tabBody">
      <div className="tabIntro"><ShieldCheck size={18} /><div><h3>Data quality intelligence</h3>
        <p>Score = Σ weight × dimension score, where each dimension score is 100 × (1 − min(1, k × defect rate)). Weights: completeness 30%, uniqueness 20%, validity 20%, consistency 15%, outliers 15%.</p></div></div>
      <div className="qualityTop">
        <div className="gaugeBox"><Gauge score={q.score} grade={q.grade} /><div className="muted">{q.summary}</div></div>
        <div className="dimBars">{dims.map((d) => (
          <div className="barRow" key={d.dimension}><span className="barName">{d.dimension} <small>({d.weight * 100}%)</small></span>
            <div className="barTrack"><div className="barFill" style={{ width: `${d.score}%`, background: d.score >= 90 ? '#34d399' : d.score >= 75 ? '#fbbf24' : '#f87171' }} /></div>
            <span className="barVal">{fmt(d.score, 1)}</span></div>))}</div>
      </div>
      <DataTable rows={q.issues.map((i) => ({ '#': i.id, severity: i.severity, dimension: i.dimension, issue: i.issue, recommended_treatment: i.recommendation }))}
        title={`Issues and recommended treatments (${q.issues.length})`} max={15} wide csvName="data_quality_issues" />
      <div className="cleanBox">
        <div className="blockTitle"><Wand2 size={12} /> Automatic cleaning (choose actions; every change is logged and reversible)</div>
        <div className="actionGrid">{q.actions.all.map((a) => (
          <label key={a} className={'action ' + (sel.includes(a) ? 'on' : '')}><input type="checkbox" checked={sel.includes(a)} onChange={() => toggle(a)} />{ACTION_LABEL[a] || a}</label>))}</div>
        <div className="btnRow">
          <button className="primary" onClick={clean} disabled={busy || !sel.length}><Wand2 size={14} />{busy ? 'Cleaning…' : 'Apply cleaning'}</button>
          {(q.cleaned || cmp) && <button className="ghost" onClick={revert} disabled={busy}><RotateCcw size={14} />Revert to original</button>}
          <button className="ghost" onClick={() => download(`/api/session/${session}/download`, 'dataset.csv')}><Download size={14} />Download current data (CSV)</button>
        </div>
        {err && <ErrorBox error={err} />}
      </div>
      {cmp && (
        <div className="compare">
          <div className="blockTitle">Before / after</div>
          <div className="kpiRow">
            <div className="kpi"><span>Quality score</span><b>{fmt(cmp.comparison.score_before, 1)} → {fmt(cmp.comparison.score_after, 1)}</b><small>{cmp.comparison.score_change >= 0 ? '+' : ''}{fmt(cmp.comparison.score_change, 1)} points</small></div>
            <div className="kpi"><span>Rows</span><b>{fmt(cmp.comparison.rows_before)} → {fmt(cmp.comparison.rows_after)}</b></div>
            <div className="kpi"><span>Missing cells</span><b>{fmt(cmp.comparison.missing_before)} → {fmt(cmp.comparison.missing_after)}</b></div>
            <div className="kpi"><span>Duplicate rows</span><b>{fmt(cmp.comparison.duplicates_before)} → {fmt(cmp.comparison.duplicates_after)}</b></div>
            <div className="kpi"><span>Issues</span><b>{cmp.comparison.issues_before} → {cmp.comparison.issues_after}</b></div>
          </div>
          <Chart chart={{ type: 'bar', title: 'Dimension scores before and after cleaning', x: 'dimension', y: ['before', 'after'], data: cmp.comparison.dimensions }} />
          <DataTable rows={cmp.log} title="Cleaning log" max={30} csvName="cleaning_log" />
          <DataTable rows={cmp.comparison.numeric_columns} title="Numeric columns before / after" max={15} wide />
        </div>
      )}
      {!cmp && q.cleaning_log?.length > 0 && <DataTable rows={q.cleaning_log} title="Cleaning already applied to this dataset" max={20} />}
    </div>
  );
}
