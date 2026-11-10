import React, { useEffect, useState } from 'react';
import { History } from 'lucide-react';
import { DataTable } from '../ResultView';
import { getJson } from '../api';
import { Loading, ErrorBox } from './common';

export default function HistoryTab({ session, version }) {
  const [h, setH] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { getJson(`/api/session/${session}/history`).then(setH).catch(setErr); }, [session, version]);
  if (err) return <ErrorBox error={err} />;
  if (!h) return <Loading text="Loading history…" />;
  const rows = (h.routing || []).map((r) => ({ time: r.at, question: r.question, planner: r.planner, status: r.status, operation: r.operation, target: r.target || r.planned_target,
    group_by: (r.group_by || []).join(', '), filters: (r.filters || []).join('; '), model: r.model, machine_learning: r.ml, seconds: r.seconds, error: r.error }));
  return (
    <div className="tabBody">
      <div className="tabIntro"><History size={18} /><div><h3>Reproducible analysis history</h3>
        <p>Every question with the planner that interpreted it, the operation and columns it was routed to, the model used and the time taken. Download it as CSV for audit or evaluation.</p></div></div>
      {rows.length ? <DataTable rows={rows} title="Routing log (this session)" max={100} wide csvName="udi_routing_log" /> : <p className="muted">No questions asked yet.</p>}
      {h.history?.length > 0 && <DataTable rows={h.history.map((x) => ({ ...x, created: new Date(x.created * 1000).toLocaleString() }))} title="Stored results" max={50} wide csvName="udi_history" />}
    </div>
  );
}
