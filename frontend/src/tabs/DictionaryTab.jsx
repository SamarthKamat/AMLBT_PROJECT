import React, { useEffect, useMemo, useState } from 'react';
import { GitBranch, Network, Sigma, KeyRound, CalendarClock, Search } from 'lucide-react';
import Chart from '../Charts';
import { DataTable } from '../ResultView';
import { getJson } from '../api';
import { fmt } from '../format';
import { Loading, ErrorBox } from './common';

export default function DictionaryTab({ session, version }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [focus, setFocus] = useState('');
  const [q, setQ] = useState('');
  useEffect(() => {
    setData(null); setErr(null);
    getJson(`/api/session/${session}/dictionary`).then(setData).catch(setErr);
  }, [session, version]);
  const pairs = useMemo(() => (data?.pairs || []).filter((p) => !focus || p.column_1 === focus || p.column_2 === focus), [data, focus]);
  if (err) return <ErrorBox error={err} />;
  if (!data) return <Loading text="Mapping how every column relates to every other column…" />;
  const dict = data.dictionary.filter((r) => !q || JSON.stringify(r).toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="tabBody">
      <div className="tabIntro"><Network size={18} /><div><h3>Column relationships & data dictionary</h3>
        <p>Every statement below is computed from the data: association strength (Pearson/Spearman for numbers, η for number–category, bias-corrected Cramér's V for category–category), hierarchies, derived formulas, date rules and keys.</p></div></div>
      <div className="findGrid">
        {data.findings.map((f, i) => <div className="finding" key={i}>{f}</div>)}
      </div>
      <div className="twoCol">
        <div>{data.heatmap && <Chart chart={data.heatmap} />}</div>
        <div>
          <div className="blockHead"><div className="blockTitle">Strongest relationships</div>
            <select value={focus} onChange={(e) => setFocus(e.target.value)} className="select">
              <option value="">all columns</option>{data.dictionary.map((r) => <option key={r.column} value={r.column}>{r.column}</option>)}
            </select></div>
          <div className="pairList">
            {pairs.slice(0, 18).map((p, i) => (
              <div className="pair" key={i}>
                <span className="pairNames">{p.column_1} <em>↔</em> {p.column_2}</span>
                <div className="barTrack"><div className={'barFill ' + (p.signed < 0 ? 'neg' : '')} style={{ width: `${Math.min(100, p.strength * 100)}%` }} /></div>
                <span className="pairVal" title={p.measure}>{p.signed !== null && p.signed !== undefined ? fmt(p.signed, 2) : fmt(p.strength, 2)}</span>
                <span className="pairMeasure">{p.measure.replace('_', ' ')} · {p.label}</span>
              </div>
            ))}
            {!pairs.length && <div className="muted">No relationships found.</div>}
          </div>
        </div>
      </div>
      <div className="struct">
        {data.hierarchies.length > 0 && <div className="structCard"><div className="blockTitle"><GitBranch size={12} /> Hierarchies (finer → coarser)</div>
          {data.hierarchies.map((h, i) => <div key={i} className="structRow"><b>{h.child}</b> → <b>{h.parent}</b> <span className="muted">{fmt(h.child_values)} → {fmt(h.parent_values)} values · {fmt(h.purity * 100, 1)}% consistent</span></div>)}</div>}
        {data.derived.length > 0 && <div className="structCard"><div className="blockTitle"><Sigma size={12} /> Derived columns</div>
          {data.derived.map((d, i) => <div key={i} className="structRow"><code>{d.formula}</code> <span className="muted">holds on {fmt(d.rows_matching * 100, 1)}% of rows</span></div>)}</div>}
        {data.date_rules.length > 0 && <div className="structCard"><div className="blockTitle"><CalendarClock size={12} /> Date rules</div>
          {data.date_rules.map((d, i) => <div key={i} className="structRow">{d.statement}</div>)}</div>}
        {(data.candidate_keys.length > 0 || data.duplicates.length > 0) && <div className="structCard"><div className="blockTitle"><KeyRound size={12} /> Keys & redundancy</div>
          {data.candidate_keys.length > 0 && <div className="structRow">Candidate key(s): <b>{data.candidate_keys.join(', ')}</b></div>}
          {data.duplicates.map((d, i) => <div key={i} className="structRow">{d.statement}</div>)}</div>}
      </div>
      <div className="blockHead"><div className="blockTitle">Data dictionary ({data.dictionary.length} columns)</div>
        <label className="search"><Search size={13} /><input placeholder="filter columns…" value={q} onChange={(e) => setQ(e.target.value)} /></label></div>
      <DataTable rows={dict.map((r) => ({ column: r.column, role: r.role, semantic_type: r.semantic_type, additive: r.additive, 'missing_%': r['missing_%'], unique: r.unique_values,
        description: r.description, strongest_relationships: r.strongest_relationships, functions: r.business_functions }))} max={60} wide csvName="data_dictionary" />
    </div>
  );
}
