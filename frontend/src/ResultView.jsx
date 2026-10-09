import React, { useState } from 'react';
import { ChevronDown, AlertTriangle, Info } from 'lucide-react';
import Chart from './Charts';
import { fmt, label } from './format';

const OP_NAMES = {
  get_summary: 'Summary', get_current_state: 'Current state', compare_groups: 'Group comparison', analyze_trend: 'Trend analysis',
  detect_anomalies: 'Anomaly detection', calculate_correlation: 'Correlation', predict_class: 'Classification model',
  predict_value: 'Regression model', forecast: 'Forecast', run_scenario: 'Scenario simulation',
};

export const opName = (op) => OP_NAMES[op] || op;

function DataTable({ rows, title, max = 12, wide = false }) {
  if (!rows?.length) return null;
  const keys = Object.keys(rows[0]).slice(0, wide ? 12 : 9);
  return (
    <div className="tableBlock">
      {title && <div className="blockTitle">{title}</div>}
      <div className="tableWrap">
        <table>
          <thead><tr>{keys.map((k) => <th key={k}>{label(k)}</th>)}</tr></thead>
          <tbody>
            {rows.slice(0, max).map((row, i) => (
              <tr key={i} className={row.selected ? 'selectedRow' : ''}>
                {keys.map((k) => <td key={k}>{fmt(row[k], 3)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > max && <div className="tableNote">Showing {max} of {rows.length} rows.</div>}
    </div>
  );
}

function Metrics({ metrics }) {
  const entries = Object.entries(metrics || {}).slice(0, 14);
  if (!entries.length) return null;
  return (
    <div className="metrics">
      {entries.map(([k, v]) => (
        <div className="metric" key={k} title={String(v)}>
          <span>{label(k)}</span>
          <b className={typeof v === 'string' && v.length > 18 ? 'small' : ''}>{fmt(v, 3)}</b>
        </div>
      ))}
    </div>
  );
}

function Confusion({ cm }) {
  if (!cm?.matrix?.length) return null;
  const max = Math.max(...cm.matrix.flat(), 1);
  return (
    <div className="tableBlock">
      <div className="blockTitle">Confusion matrix (held-out rows; rows = actual, columns = predicted)</div>
      <div className="tableWrap">
        <table className="cm">
          <thead><tr><th /> {cm.labels.map((l) => <th key={l}>{l}</th>)}</tr></thead>
          <tbody>
            {cm.matrix.map((row, i) => (
              <tr key={i}>
                <th>{cm.labels[i]}</th>
                {row.map((v, j) => <td key={j} style={{ background: `rgba(${i === j ? '52,211,153' : '248,113,113'},${0.08 + 0.5 * (v / max)})` }}>{v}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Prediction({ p }) {
  if (!p) return null;
  const given = Object.entries(p.inputs_given || {});
  const assumed = Object.entries(p.inputs_assumed || {});
  return (
    <div className="callout">
      <div className="blockTitle">Prediction ({p.basis})</div>
      {p.class !== undefined ? (
        <>
          <div className="big">{String(p.class)}</div>
          <div className="probs">{(p.probabilities || []).map((x) => <span key={x.class}>{x.class}: {fmt(x.probability * 100, 1)}%</span>)}</div>
        </>
      ) : (
        <>
          <div className="big">{fmt(p.value)} <small>{p.target}</small></div>
          {p.interval_80 && <div className="probs"><span>80% range {fmt(p.interval_80[0])} – {fmt(p.interval_80[1])}</span><span>typical error (RMSE) {fmt(p.typical_error_rmse)}</span></div>}
        </>
      )}
      {given.length > 0 && <div className="kv"><b>You specified:</b> {given.map(([k, v]) => `${k} = ${fmt(v)}`).join(' · ')}</div>}
      {assumed.length > 0 && <div className="kv"><b>Typical values used for the rest:</b> {assumed.map(([k, v]) => `${k} = ${fmt(v)}`).join(' · ')}</div>}
    </div>
  );
}

function Scenario({ s }) {
  if (!s) return null;
  return (
    <div className="callout">
      <div className="blockTitle">Scenario ({s.model})</div>
      <div className="scenarioRow">
        <div><span>Baseline</span><b>{fmt(s.baseline)}</b></div>
        <div className="arrow">→</div>
        <div><span>Scenario</span><b>{fmt(s.scenario)}</b></div>
        <div className={'delta ' + (s.change >= 0 ? 'up' : 'down')}><span>Change</span><b>{s.change >= 0 ? '+' : ''}{fmt(s.change)} ({s.change_pct >= 0 ? '+' : ''}{fmt(s.change_pct, 1)}%)</b></div>
      </div>
      <div className="kv"><b>Changes applied:</b> {(s.changes || []).map((c) => `${c.column}: ${c.type === 'percent' ? (c.value >= 0 ? '+' : '') + c.value + '%' : c.type === 'absolute' ? (c.value >= 0 ? '+' : '') + c.value : 'set to ' + c.value}`).join(' · ')}</div>
      <div className="kv warn"><Info size={13} /> Model-based estimate; it does not establish causal impact.</div>
    </div>
  );
}

function Importance({ ex }) {
  if (!ex?.items?.length) return null;
  const top = Math.max(...ex.items.map((i) => i['share_%']), 1);
  return (
    <div className="tableBlock">
      <div className="blockTitle">What the model relies on (feature importance)</div>
      <div className="bars">
        {ex.items.slice(0, 10).map((i) => (
          <div className="barRow" key={i.feature}>
            <span className="barName">{i.feature}</span>
            <div className="barTrack"><div className="barFill" style={{ width: `${(i['share_%'] / top) * 100}%` }} /></div>
            <span className="barVal">{fmt(i['share_%'], 1)}%</span>
          </div>
        ))}
      </div>
      <div className="tableNote">{ex.method}. {ex.note}</div>
    </div>
  );
}

function Notes({ items, kind, icon }) {
  if (!items?.length) return null;
  return (
    <div className={'notes ' + kind}>
      {items.map((t, i) => <p key={i}>{icon}<span>{t}</span></p>)}
    </div>
  );
}

function Details({ plan, result }) {
  const [open, setOpen] = useState(false);
  const prov = result?.provenance || {};
  const ev = result?.evaluation || {};
  const rows = [
    ['Operation', plan?.operation ? `${plan.operation} (${opName(plan.operation)})` : '—'],
    ['Target', plan?.target_column],
    ['Features', ev.features_used?.join(', ')],
    ['Group by', plan?.group_by?.join(', ')],
    ['Time column', prov.time_column],
    ['Aggregation', plan?.aggregation],
    ['Filters', prov.filters?.join('; ')],
    ['Selected model', ev.selected_model],
    ['Used machine learning', prov.uses_machine_learning ? 'Yes' : 'No — deterministic Python / statistics'],
    ['Rows analysed', prov.rows_analysed !== undefined ? `${fmt(prov.rows_analysed)} of ${fmt(prov.rows_in_dataset)}` : null],
    ['Reused cached model', ev.cached_model !== undefined ? (ev.cached_model ? 'Yes' : 'No — trained for this question') : null],
    ['Compute time', prov.seconds !== undefined ? `${fmt(prov.seconds)} s` : null],
    ['Planner reasoning', plan?.reasoning],
  ].filter(([, v]) => v !== null && v !== undefined && v !== '');
  return (
    <>
      <button className="details" onClick={() => setOpen(!open)}>Analysis details <ChevronDown size={15} className={open ? 'rot' : ''} /></button>
      {open && <div className="detailsBox">{rows.map(([k, v]) => <div key={k}><b>{k}</b><span>{v}</span></div>)}</div>}
    </>
  );
}

export default function ResultView({ result, plan }) {
  if (!result) return null;
  const ev = result.evaluation;
  const t = result.tables || {};
  return (
    <div className="result">
      <div className="resultHead">
        <span className="opBadge">{opName(result.operation)}</span>
        {result.provenance?.uses_machine_learning && ev?.selected_model && <span className="opBadge ml">{ev.selected_model}</span>}
        {result.provenance && !result.provenance.uses_machine_learning && <span className="opBadge plain">No ML needed</span>}
        <h4>{result.title}</h4>
      </div>

      <Metrics metrics={result.metrics} />
      <Prediction p={result.prediction} />
      <Scenario s={result.scenario} />

      {(result.charts || []).map((c, i) => <Chart key={i} chart={c} />)}

      <DataTable rows={result.table} title={result.table_title} wide />
      {ev?.confusion_matrix && <Confusion cm={ev.confusion_matrix} />}
      <DataTable rows={t.per_class} title="Per-class results (held-out rows)" />
      <Importance ex={result.explainability} />
      {t.candidate_models?.length > 0 && <DataTable rows={t.candidate_models} title={`Candidate models compared (selected by ${ev?.selection_metric || 'validation score'})`} wide />}
      {ev?.baseline?.model && <div className="tableNote">Best simple baseline: {ev.baseline.model} (RMSE {fmt(ev.baseline.RMSE, 3)}){ev['skill_vs_baseline_%'] > 0 ? `; selected model error is ${fmt(ev['skill_vs_baseline_%'], 1)}% lower` : ''}.</div>}
      {ev?.baseline?.description && <div className="tableNote">Baseline: {ev.baseline.description}{ev.baseline.accuracy !== undefined ? ` → accuracy ${fmt(ev.baseline.accuracy, 3)}` : ''}{ev.baseline.RMSE !== undefined ? ` → RMSE ${fmt(ev.baseline.RMSE, 3)}` : ''}.</div>}
      {Object.entries(t).filter(([k]) => !['candidate_models', 'per_class', 'feature_importance'].includes(k)).map(([k, rows]) => <DataTable key={k} rows={rows} title={label(k)} />)}

      <Notes items={result.warnings} kind="warn" icon={<AlertTriangle size={13} />} />
      <Notes items={result.limitations} kind="limits" icon={<Info size={13} />} />
      {result.assumptions?.length > 0 && <Notes items={result.assumptions.map((a) => `Assumption: ${a}`)} kind="assume" icon={<Info size={13} />} />}
      <Details plan={plan} result={result} />
    </div>
  );
}
