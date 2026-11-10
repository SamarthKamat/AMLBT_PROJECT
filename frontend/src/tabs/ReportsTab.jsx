import React, { useEffect, useRef, useState } from 'react';
import { FileText, FileSpreadsheet, FileJson, Play } from 'lucide-react';
import { ReportBody } from '../ResultView';
import { download, getJson, postJson } from '../api';
import { Loading, ErrorBox } from './common';

const FUNCTIONS = [
  ['sales', 'Sales', 'Revenue, volumes, regions, products, margins, trend and forecast'],
  ['marketing', 'Marketing', 'Conversion / churn drivers, segments, campaign and customer mix'],
  ['hr', 'HR', 'Attrition drivers, pay and satisfaction by role, workforce segments'],
  ['operations', 'Operations', 'Demand, capacity, lead times, anomalies and forecasts'],
  ['finance', 'Finance', 'Revenue, cost, profit, margins, loss-makers, root cause'],
  ['customer', 'Customer', 'Churn, tenure, value and customer segments'],
  ['general', 'General', 'Overview of KPIs, drivers, quality and relationships'],
];

export default function ReportsTab({ session, version, columns }) {
  const [fn, setFn] = useState('sales');
  const [target, setTarget] = useState('');
  const [job, setJob] = useState(null);
  const [report, setReport] = useState(null);
  const [err, setErr] = useState(null);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => { setReport(null); setJob(null); }, [session, version]);

  const poll = (id) => {
    getJson(`/api/jobs/${id}`).then((j) => {
      setJob(j);
      if (j.status === 'done') setReport(j.result);
      else if (j.status === 'error') setErr(Object.assign(new Error(j.error?.error || 'Report failed'), { code: j.error?.code }));
      else timer.current = setTimeout(() => poll(id), 1200);
    }).catch(setErr);
  };
  const start = async () => {
    setErr(null); setReport(null);
    try { const r = await postJson(`/api/session/${session}/report`, { function: fn, target: target || null }); setJob({ id: r.job_id, status: 'queued', progress: 0, message: 'queued' }); poll(r.job_id); }
    catch (e) { setErr(e); }
  };
  const running = job && (job.status === 'queued' || job.status === 'running');
  return (
    <div className="tabBody">
      <div className="tabIntro"><FileText size={18} /><div><h3>Automated business reports</h3>
        <p>A Data Analyst, a Business Analyst, an ML and a Report agent run UDI's validated tools and assemble KPIs, findings, recommended actions (with evidence) and charts. Export as PDF, Excel or JSON.</p></div></div>
      <div className="fnGrid">{FUNCTIONS.map(([id, name, desc]) => (
        <button key={id} className={'fnCard ' + (fn === id ? 'on' : '')} onClick={() => setFn(id)}><b>{name}</b><small>{desc}</small></button>))}</div>
      <div className="btnRow">
        <select className="select" value={target} onChange={(e) => setTarget(e.target.value)} title="Optional: the KPI or outcome the report should focus on">
          <option value="">Focus: automatic</option>{columns.map((c) => <option key={c} value={c}>Focus: {c}</option>)}
        </select>
        <button className="primary" onClick={start} disabled={running}><Play size={14} />{running ? 'Generating…' : 'Generate report'}</button>
        {report && <>
          <button className="ghost" onClick={() => download(`/api/jobs/${job.id}/download?format=pdf`, 'report.pdf')}><FileText size={14} />PDF</button>
          <button className="ghost" onClick={() => download(`/api/jobs/${job.id}/download?format=xlsx`, 'report.xlsx')}><FileSpreadsheet size={14} />Excel</button>
          <button className="ghost" onClick={() => download(`/api/jobs/${job.id}/download?format=json`, 'report.json')}><FileJson size={14} />JSON</button>
        </>}
      </div>
      {running && <Loading text={job.message || 'Working…'} progress={job.progress} />}
      {err && <ErrorBox error={err} />}
      {report && <ReportBody report={report} />}
    </div>
  );
}
