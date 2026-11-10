import React, { useState } from 'react';
import { ChevronDown, AlertTriangle, Info, Lightbulb, Target, Download, Cpu, CheckCircle2, CircleSlash, Sparkles } from 'lucide-react';
import Chart from './Charts';
import { fmt, label } from './format';
import { saveText, toCsv } from './api';

const OP_NAMES = {
  get_summary: 'Summary', get_current_state: 'Current state', compare_groups: 'Group comparison', analyze_trend: 'Trend analysis',
  detect_anomalies: 'Anomaly detection', calculate_correlation: 'Correlation', predict_class: 'Classification model',
  predict_value: 'Regression model', forecast: 'Forecast', run_scenario: 'Scenario simulation', query_records: 'Record lookup',
  pivot_table: 'Pivot table', run_eda: 'Exploratory analysis', assess_data_quality: 'Data quality', describe_relationships: 'Column relationships',
  explain_change: 'Root-cause analysis', segment_records: 'Segmentation', generate_report: 'Business report',
};
export const opName = (op) => OP_NAMES[op] || op;

export function DataTable({ rows, title, max = 12, wide = false, csvName }) {
  const [all, setAll] = useState(false);
  if (!rows?.length) return null;
  const keys = Object.keys(rows[0]).slice(0, wide ? 12 : 9);
  const shown = all ? rows : rows.slice(0, max);
  return (
    <div className="tableBlock">
      <div className="blockHead">
        {title && <div className="blockTitle">{title}</div>}
        <button className="mini" onClick={() => saveText(toCsv(rows), `${csvName || title || 'table'}.csv`.replace(/[^\w.-]+/g, '_'))} title="Download as CSV"><Download size={12} />CSV</button>
      </div>
      <div className="tableWrap">
        <table>
          <thead><tr>{keys.map((k) => <th key={k}>{label(k)}</th>)}</tr></thead>
          <tbody>
            {shown.map((row, i) => (
              <tr key={i} className={row.selected ? 'selectedRow' : ''}>
                {keys.map((k) => <td key={k} className={typeof row[k] === 'string' && row[k].length > 40 ? 'wrap' : ''}>{fmt(row[k], 3)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > max && <button className="linkBtn" onClick={() => setAll(!all)}>{all ? 'Show fewer' : `Show all ${rows.length} rows`}</button>}
    </div>
  );
}

export function Metrics({ metrics }) {
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
              <tr key={i}><th>{cm.labels[i]}</th>
                {row.map((v, j) => <td key={j} style={{ background: `rgba(${i === j ? '52,211,153' : '248,113,113'},${0.08 + 0.5 * (v / max)})` }}>{v}</td>)}</tr>
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
      {assumed.length > 0 && <div className="kv"><b>Typical values used for the rest:</b> {assumed.slice(0, 10).map(([k, v]) => `${k} = ${fmt(v)}`).join(' · ')}</div>}
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

function Explainability({ ex }) {
  if (!ex) return null;
  const sh = ex.shap?.items?.length ? ex.shap : null;
  const items = sh ? sh.items : ex.items || [];
  if (!items.length) return null;
  const top = Math.max(...items.map((i) => i['share_%']), 1);
  return (
    <div className="tableBlock">
      <div className="blockTitle">{sh ? 'Explainable AI: SHAP feature impact' : 'What the model relies on (feature importance)'}</div>
      <div className="bars">
        {items.slice(0, 10).map((i) => (
          <div className="barRow" key={i.feature} title={i.direction || ''}>
            <span className="barName">{i.feature}</span>
            <div className="barTrack"><div className="barFill" style={{ width: `${(i['share_%'] / top) * 100}%` }} /></div>
            <span className="barVal">{fmt(i['share_%'], 1)}%</span>
            {i.direction && <span className="barDir">{i.direction}</span>}
          </div>
        ))}
      </div>
      {ex.local?.summary && <div className="kv"><b>This prediction:</b> {ex.local.summary} <span className="muted">(base {fmt(ex.local.base_value)} → prediction {fmt(ex.local.prediction)}; additivity gap {fmt(ex.local.additivity_gap, 4)})</span></div>}
      <div className="tableNote">{sh ? `${sh.method}. ${sh.note}` : `${ex.method}. ${ex.note}`}</div>
    </div>
  );
}

function ModelSelection({ ev }) {
  if (!ev?.selection_rationale) return null;
  const tv = ev.temporal_validation;
  return (
    <div className="callout slim">
      <div className="blockTitle"><Cpu size={12} /> Why this model</div>
      <div className="kv">{ev.selection_rationale}</div>
      {ev.p_value_vs_runner_up !== null && ev.p_value_vs_runner_up !== undefined && <div className="kv">Paired fold test vs runner-up ({ev.runner_up}): p = {fmt(ev.p_value_vs_runner_up, 3)}.</div>}
      {tv && <div className="kv">Time-ordered check (train to {tv.train_until}, test from {tv.test_from}): {Object.entries(tv.metrics).map(([k, v]) => `${k} ${fmt(v, 3)}`).join(' · ')}.</div>}
      {ev.leakage_suspects?.length > 0 && <div className="kv warn"><AlertTriangle size={13} /> Possible leakage: {ev.leakage_suspects.join(', ')}.</div>}
    </div>
  );
}

export function BusinessInsights({ items, title = 'Business findings and suggested actions' }) {
  if (!items?.length) return null;
  return (
    <div className="bizBox">
      <div className="blockTitle"><Lightbulb size={12} /> {title}</div>
      {items.map((r, i) => (
        <div className={'biz ' + (r.priority || 'medium')} key={i}>
          <div className="bizFinding"><span className={'prio ' + (r.priority || 'medium')}>{r.priority || 'medium'}</span>{r.finding}</div>
          <div className="bizAction"><Target size={12} /> {r.action}</div>
          <div className="bizEvidence">Evidence: {r.evidence}{r.impact_estimate ? ` · ${r.impact_estimate}` : ''}</div>
        </div>
      ))}
    </div>
  );
}

function Insights({ items }) {
  if (!items?.length) return null;
  return (
    <div className="insights">
      <div className="blockTitle"><Sparkles size={12} /> Key insights</div>
      <ul>{items.map((t, i) => <li key={i}>{t}</li>)}</ul>
    </div>
  );
}

function Notes({ items, kind, icon }) {
  if (!items?.length) return null;
  return <div className={'notes ' + kind}>{items.map((t, i) => <p key={i}>{icon}<span>{t}</span></p>)}</div>;
}

export function AgentTrace({ steps, title = 'Agent trace' }) {
  const [open, setOpen] = useState(false);
  if (!steps?.length) return null;
  return (
    <>
      <button className="details" onClick={() => setOpen(!open)}>{title} ({steps.length} steps) <ChevronDown size={15} className={open ? 'rot' : ''} /></button>
      {open && (
        <div className="trace">
          {steps.map((s, i) => (
            <div className="traceRow" key={i}>
              {s.status === 'ok' ? <CheckCircle2 size={13} className="ok" /> : <CircleSlash size={13} className="skip" />}
              <b>{s.agent}</b><code>{s.tool}</code><span className="muted">{fmt(s.seconds)} s</span>
              <div className="traceHead">{s.headline}</div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function Details({ plan, result }) {
  const [open, setOpen] = useState(false);
  const prov = result?.provenance || {};
  const ev = result?.evaluation || {};
  const rows = [
    ['Operation', plan?.operation ? `${plan.operation} (${opName(plan.operation)})` : '—'],
    ['Target', plan?.target_column], ['Features', ev.features_used?.join(', ')], ['Group by', plan?.group_by?.join(', ')],
    ['Pivot column', plan?.pivot_column], ['Time column', prov.time_column], ['Aggregation', plan?.aggregation], ['Filters', prov.filters?.join('; ')],
    ['Selected model', ev.selected_model], ['Used machine learning', prov.uses_machine_learning ? 'Yes' : 'No — deterministic Python / statistics'],
    ['Rows analysed', prov.rows_analysed !== undefined ? `${fmt(prov.rows_analysed)} of ${fmt(prov.rows_in_dataset)}` : null],
    ['Reused cached model', ev.cached_model !== undefined ? (ev.cached_model ? 'Yes' : 'No — trained for this question') : null],
    ['Compute time', prov.seconds !== undefined ? `${fmt(prov.seconds)} s` : null], ['Planner reasoning', plan?.reasoning],
  ].filter(([, v]) => v !== null && v !== undefined && v !== '');
  return (
    <>
      <button className="details" onClick={() => setOpen(!open)}>Analysis details <ChevronDown size={15} className={open ? 'rot' : ''} /></button>
      {open && <div className="detailsBox">{rows.map(([k, v]) => <div key={k}><b>{k}</b><span>{v}</span></div>)}</div>}
    </>
  );
}

export function ReportBody({ report, compact = false }) {
  if (!report) return null;
  return (
    <div className="report">
      <div className="reportHead">
        <div><span className="eyebrow">{report.function_label} report</span><h3>{report.title}</h3>
          <div className="muted">{report.generated_at} · {fmt(report.dataset.rows)} rows × {report.dataset.columns} columns · data quality {fmt(report.data_quality.score, 0)}/100 (grade {report.data_quality.grade}) · {fmt(report.seconds)} s</div></div>
      </div>
      <div className="execSum"><div className="blockTitle">Executive summary</div><ul>{report.executive_summary.map((b, i) => <li key={i}>{b}</li>)}</ul></div>
      <div className="kpiRow">{report.kpis.map((k) => <div className="kpi" key={k.label}><span>{k.label}</span><b>{fmt(k.value)}{k.unit || ''}</b><small>{k.note}</small></div>)}</div>
      <BusinessInsights items={report.recommendations} title="Findings and recommended management actions" />
      {(compact ? report.sections.slice(0, 3) : report.sections).map((s, i) => (
        <section className="repSection" key={i}>
          <div className="repTitle"><h4>{s.title}</h4><span className="agentTag">{s.agent}</span></div>
          <p>{s.summary}</p>
          {s.insights?.length > 0 && <ul className="small">{s.insights.map((t, j) => <li key={j}>{t}</li>)}</ul>}
          <div className={s.charts.length > 1 ? 'chartGrid' : ''}>{s.charts.map((c, j) => <Chart key={j} chart={c} />)}</div>
          {!compact && <DataTable rows={s.table} title={s.table_title} max={8} />}
        </section>
      ))}
      {compact && report.sections.length > 3 && <div className="tableNote">{report.sections.length - 3} more section(s) in the Reports tab and in the PDF / Excel export.</div>}
      {!compact && (
        <>
          <section className="repSection"><div className="repTitle"><h4>Data quality</h4><span className="agentTag">Data Analyst Agent</span></div><p>{report.data_quality.summary}</p>
            <DataTable rows={report.data_quality.top_issues} title="Top issues" max={6} /></section>
          <section className="repSection"><div className="repTitle"><h4>How the columns relate</h4><span className="agentTag">Data Analyst Agent</span></div>
            <ul className="small">{report.relationships.map((t, i) => <li key={i}>{t}</li>)}</ul></section>
          <Notes items={report.limitations} kind="limits" icon={<Info size={13} />} />
        </>
      )}
      <AgentTrace steps={report.agent_trace} title="Multi-agent trace" />
    </div>
  );
}

export default function ResultView({ result, plan }) {
  if (!result) return null;
  const ev = result.evaluation;
  const t = result.tables || {};
  if (result.report) {
    return (
      <div className="result">
        <div className="resultHead"><span className="opBadge">{opName(result.operation)}</span><span className="opBadge ml">multi-agent</span></div>
        <ReportBody report={result.report} compact />
      </div>
    );
  }
  const hiddenTables = ['candidate_models', 'per_class', 'feature_importance', 'data_dictionary'];
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
      <Insights items={result.insights} />
      <div className={(result.charts || []).length > 2 ? 'chartGrid' : ''}>
        {(result.charts || []).map((c, i) => <Chart key={i} chart={c} preferred={i === 0 ? result.preferred_chart : undefined} />)}
      </div>
      <BusinessInsights items={result.business_insights} />
      <ModelSelection ev={ev} />
      <DataTable rows={result.table} title={result.table_title} wide />
      {ev?.confusion_matrix && <Confusion cm={ev.confusion_matrix} />}
      <DataTable rows={t.per_class} title="Per-class results (held-out rows)" />
      <Explainability ex={result.explainability} />
      {t.candidate_models?.length > 0 && <DataTable rows={t.candidate_models} title={`Candidate models compared (selected by ${ev?.selection_metric || 'validation score'})`} wide />}
      {ev?.baseline?.model && <div className="tableNote">Best simple baseline: {ev.baseline.model} (RMSE {fmt(ev.baseline.RMSE, 3)}){ev['skill_vs_baseline_%'] > 0 ? `; selected model error is ${fmt(ev['skill_vs_baseline_%'], 1)}% lower` : ''}.</div>}
      {ev?.baseline?.description && <div className="tableNote">Baseline: {ev.baseline.description}{ev.baseline.accuracy !== undefined ? ` → accuracy ${fmt(ev.baseline.accuracy, 3)}` : ''}{ev.baseline.RMSE !== undefined ? ` → RMSE ${fmt(ev.baseline.RMSE, 3)}` : ''}.</div>}
      {t.data_dictionary?.length > 0 && <DataTable rows={t.data_dictionary} title="Data dictionary" wide max={8} />}
      {Object.entries(t).filter(([k]) => !hiddenTables.includes(k)).map(([k, rows]) => <DataTable key={k} rows={rows} title={label(k)} />)}
      <Notes items={result.warnings} kind="warn" icon={<AlertTriangle size={13} />} />
      <Notes items={result.limitations} kind="limits" icon={<Info size={13} />} />
      {result.assumptions?.length > 0 && <Notes items={result.assumptions.map((a) => `Assumption: ${a}`)} kind="assume" icon={<Info size={13} />} />}
      <Details plan={plan} result={result} />
    </div>
  );
}
