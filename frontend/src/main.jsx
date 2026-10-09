import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Upload, Send, Plus, Sparkles, Activity, Brain, ShieldCheck, ChevronDown, BarChart3, FileSpreadsheet, Bot, User, RefreshCw, AlertOctagon, HelpCircle } from 'lucide-react';
import ResultView from './ResultView';
import { apiFetch, postJson, ApiError, ERROR_INFO } from './api';
import { fmt } from './format';
import './styles.css';
import './extra.css';

const fallbackDomains = [
  { id: 'company', label: 'Company', icon: '▣', description: 'Sales, demand, inventory, customers and profitability.' },
  { id: 'hospital', label: 'Hospital', icon: '✚', description: 'Patient load, capacity, waiting time and staffing.' },
  { id: 'college', label: 'College', icon: '⌂', description: 'Attendance, students, faculty and resource utilization.' },
  { id: 'municipality', label: 'Municipality', icon: '◈', description: 'Waste, complaints, service demand and zones.' },
  { id: 'ngo', label: 'NGO', icon: '♡', description: 'Beneficiaries, donations, aid and volunteer demand.' },
  { id: 'community', label: 'Community', icon: '⌁', description: 'Local requests, participation, shortages and needs.' },
  { id: 'event_organizer', label: 'Event Organizer', icon: '★', description: 'Attendance, capacity, registrations and resources.' },
];

function statusView(health) {
  if (!health) return { cls: '', text: 'Checking backend…', title: '' };
  if (health.backend_offline) return { cls: 'bad', text: 'Backend offline', title: 'Start Flask: python app.py (in /backend)' };
  const g = health.gemini || {};
  if (g.status === 'ok') return { cls: 'live', text: `Gemini ready · ${g.model}`, title: g.detail };
  if (g.status === 'configured') return { cls: 'live', text: `Gemini configured · ${g.model}`, title: 'Key and model are set. Click to verify the connection.' };
  if (g.status === 'no_api_key') return { cls: 'warn', text: 'Gemini key missing', title: g.detail };
  if (g.status === 'no_model') return { cls: 'warn', text: 'Gemini model not set', title: g.detail };
  if (g.status === 'model_unavailable') return { cls: 'bad', text: `Gemini model unavailable · ${g.model}`, title: `${g.detail}${g.available_models ? ' Available: ' + g.available_models.join(', ') : ''}` };
  if (g.status === 'sdk_missing') return { cls: 'bad', text: 'google-genai not installed', title: g.detail };
  return { cls: 'bad', text: 'Gemini unreachable', title: g.detail };
}

function App() {
  const [domains, setDomains] = useState(fallbackDomains);
  const [domain, setDomain] = useState('company');
  const [session, setSession] = useState('');
  const [profile, setProfile] = useState(null);
  const [showCols, setShowCols] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [health, setHealth] = useState(null);
  const fileRef = useRef(null);
  const endRef = useRef(null);

  const checkHealth = useCallback(async (live = false) => {
    try { setHealth(await apiFetch('/api/health' + (live ? '?check=1' : ''), {}, 30000)); } catch { setHealth({ backend_offline: true }); }
  }, []);

  useEffect(() => {
    apiFetch('/api/domains').then((d) => Array.isArray(d) && setDomains(d)).catch(() => {});
    checkHealth(false);
    postJson('/api/session', { domain: 'company' }).then((d) => d?.session_id && setSession(d.session_id)).catch(() => {});
  }, [checkHealth]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages]);

  const current = domains.find((d) => d.id === domain) || fallbackDomains[0];
  const st = statusView(health);

  const resetSession = async (nextDomain) => {
    try {
      const data = await postJson('/api/new-chat', { session_id: session, domain: nextDomain });
      setSession(data.session_id);
    } catch {
      try { const d = await postJson('/api/session', { domain: nextDomain }); setSession(d.session_id); } catch { /* shown on next request */ }
    }
    setMessages([]); setProfile(null); setInput(''); setShowCols(false);
  };

  const selectDomain = async (id) => { setDomain(id); await resetSession(id); };
  const newChat = async () => { await resetSession(domain); };

  const upload = async (file) => {
    if (!file) return;
    setUploading(true);
    const fd = new FormData();
    fd.append('file', file); fd.append('domain', domain); fd.append('session_id', session);
    try {
      const data = await apiFetch('/api/upload', { method: 'POST', body: fd });
      setSession(data.session_id);
      setProfile(data.profile);
      const p = data.profile;
      setMessages([{ role: 'assistant', content: `${file.name} is loaded: ${fmt(p.rows)} rows, ${p.columns.length} columns (${p.numeric_columns.length} numeric, ${p.categorical_columns.length} categorical, ${p.datetime_columns.length} date/time).${p.date_range ? ` Period covered: ${String(p.date_range.start).slice(0, 10)} to ${String(p.date_range.end).slice(0, 10)}.` : ''} Ask anything about the data.`, warnings: p.warnings }]);
    } catch (e) {
      setMessages([{ role: 'assistant', error: e }]);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const ask = async (question) => {
    const query = (question ?? input).trim();
    if (!query || loading || !profile) return;
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: query }, { role: 'assistant', loading: true }]);
    setLoading(true);
    try {
      const data = await postJson('/api/chat', { session_id: session, message: query });
      setMessages((prev) => [...prev.slice(0, -1), { role: 'assistant', content: data.answer, kind: data.type, options: data.options, source: data.answer_source, note: data.explanation_note, plan: data.plan, result: data.result, timing: data.timing, question: query }]);
    } catch (e) {
      setMessages((prev) => [...prev.slice(0, -1), { role: 'assistant', error: e, question: query }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="orb orb1" /><div className="orb orb2" /><div className="orb orb3" />
      <header className="topbar">
        <div className="brand">
          <div className="brandMark"><Sparkles size={20} /></div>
          <div>
            <div className="brandName">Universal Dataset Intelligence</div>
            <div className="brandSub">Conversational analytics · every number computed in Python</div>
          </div>
        </div>
        <button className="status" onClick={() => checkHealth(true)} title={st.title || 'Click to re-check'}>
          <span className={'dot ' + st.cls} />{st.text}<RefreshCw size={12} className="refresh" />
        </button>
      </header>

      <main className="shell">
        <section className="hero">
          <div className="heroTag"><span>●</span>GEMINI PLANS · PYTHON COMPUTES</div>
          <h1>Talk to your <em>data.</em><br />Make better decisions.</h1>
          <p>Upload a dataset and ask in plain language. Gemini turns the question into a structured plan, UDI validates it against your real columns, and Python runs the right operation — a plain aggregation, a trend test, anomaly detection, or a model trained on your data and chosen by validation.</p>
          <div className="heroPills">
            <span><ShieldCheck size={15} />Plans are validated against your data</span>
            <span><Brain size={15} />Models are trained per dataset, not pre-assigned</span>
            <span><Activity size={15} />Domain is context, not a model lock</span>
          </div>
        </section>

        <div className="layout">
          <aside className="sidebar">
            <div className="panel">
              <div className="panelTitle">Domain (context only)</div>
              <div className="domainList">
                {domains.map((item) => (
                  <button key={item.id} className={'domain ' + (domain === item.id ? 'active' : '')} onClick={() => selectDomain(item.id)}>
                    <span className="domainIcon">{item.icon}</span>
                    <span><b>{item.label}</b><small>{item.description}</small></span>
                  </button>
                ))}
              </div>
            </div>

            <div className="panel uploadPanel">
              <div className="panelTitle">Dataset</div>
              <button className="upload" onClick={() => fileRef.current?.click()} disabled={uploading}>
                <Upload size={18} /><span>{uploading ? 'Reading and profiling…' : 'Upload CSV / Excel'}</span>
              </button>
              <input ref={fileRef} type="file" hidden accept=".csv,.xlsx,.xls" onChange={(e) => upload(e.target.files?.[0])} />
              {profile && (
                <div className="profile">
                  <div className="profileHead"><FileSpreadsheet size={15} /><span>{profile.dataset_name}</span></div>
                  <div className="statGrid">
                    <div><b>{fmt(profile.rows)}</b><small>Rows</small></div>
                    <div><b>{profile.columns.length}</b><small>Columns</small></div>
                    <div><b>{fmt(profile.missing_total)}</b><small>Missing cells</small></div>
                    <div><b>{fmt(profile.duplicate_rows)}</b><small>Duplicate rows</small></div>
                  </div>
                  <button className="details slim" onClick={() => setShowCols(!showCols)}>Columns and detected types <ChevronDown size={14} className={showCols ? 'rot' : ''} /></button>
                  {showCols && (
                    <div className="colList">
                      {profile.columns.map((c) => (
                        <div key={c.name} title={c.note || ''}><b>{c.name}</b><span className={'role ' + c.role}>{c.role}</span></div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <button className="newChat" onClick={newChat}><Plus size={17} />New chat</button>
          </aside>

          <section className="chatArea">
            <div className="chatHeader">
              <div><span className="eyebrow">ACTIVE WORKSPACE</span><h2>{current.label} context</h2></div>
              <div className="engine"><span className={'pulse ' + (loading ? 'busy' : '')} />{loading ? 'Planning and computing…' : profile ? 'Ready' : 'Waiting for a dataset'}</div>
            </div>

            {!profile && messages.length === 0 && (
              <div className="empty">
                <div className="emptyIcon"><Sparkles size={28} /></div>
                <h3>Start with your dataset</h3>
                <p>Upload a CSV or Excel file. You do not choose a target or a model: each question is planned, checked against your columns and computed on demand.</p>
                <div className="cards">
                  <div className="feature"><Brain /><b>Ask naturally</b><span>“Which city is slowest?” “Is it getting worse?” “What if distance grows 20%?”</span></div>
                  <div className="feature"><BarChart3 /><b>Verified analysis</b><span>Python computes every number, chart and model metric.</span></div>
                  <div className="feature"><ShieldCheck /><b>Safe by design</b><span>Missing columns and thin data are reported, never guessed.</span></div>
                </div>
              </div>
            )}

            {messages.map((m, i) => <Message key={i} m={m} onAsk={ask} onRetry={() => ask(m.question)} busy={loading} />)}
            <div ref={endRef} />

            <div className="composer">
              <textarea value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } }}
                placeholder={profile ? 'Ask a question about your dataset…' : 'Upload a dataset to begin…'} disabled={!profile || loading} />
              <button onClick={() => ask()} disabled={!profile || !input.trim() || loading} aria-label="Send"><Send size={19} /></button>
              <div className="composerHint">Enter to send · Shift + Enter for a new line · follow-ups like “only for Goa” use the previous question</div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function ErrorCard({ error, onRetry, busy }) {
  const err = error instanceof ApiError ? error : new ApiError(error?.message || 'Unexpected error');
  const [title, hint] = ERROR_INFO[err.code] || ERROR_INFO.BACKEND_ERROR;
  return (
    <div className="errorCard">
      <div className="errorHead"><AlertOctagon size={16} /><b>{title}</b><code>{err.code}</code></div>
      <p>{err.message}</p>
      <p className="hint">{hint}</p>
      {err.plan?.operation && <p className="hint">Planned operation: <code>{err.plan.operation}</code>{err.plan.target_column ? <> on <code>{err.plan.target_column}</code></> : null}</p>}
      {err.retryable && onRetry && <button className="retry" onClick={onRetry} disabled={busy}><RefreshCw size={13} />Retry</button>}
    </div>
  );
}

function Message({ m, onAsk, onRetry, busy }) {
  if (m.loading) {
    return (
      <div className="msg assistant"><div className="avatar ai"><Bot size={16} /></div>
        <div className="bubble typing"><span /><span /><span /></div></div>
    );
  }
  const user = m.role === 'user';
  return (
    <div className={'msg ' + (user ? 'user' : 'assistant')}>
      <div className={'avatar ' + (user ? 'usr' : 'ai')}>{user ? <User size={16} /> : <Bot size={16} />}</div>
      <div className="messageBody">
        {m.error ? <ErrorCard error={m.error} onRetry={m.question ? onRetry : null} busy={busy} /> : (
          <div className={'bubble ' + (m.kind === 'clarification' ? 'clarify' : '')}>
            {m.kind === 'clarification' && <div className="clarifyHead"><HelpCircle size={14} />I need one detail before running this</div>}
            {m.content}
          </div>
        )}
        {m.kind === 'clarification' && m.options?.length > 0 && (
          <div className="chips">{m.options.map((o) => <button key={o} onClick={() => onAsk(o)} disabled={busy}>{o}</button>)}</div>
        )}
        {m.note && <div className="sourceNote">{m.note}</div>}
        {m.warnings?.length > 0 && m.warnings.map((w, i) => <div className="sourceNote" key={i}>{w}</div>)}
        {m.result && <ResultView result={m.result} plan={m.plan} />}
        {m.result && m.source && (
          <div className="sourceTag">{m.source === 'gemini' ? 'Wording by Gemini; every number checked against the Python result.' : 'Wording written by UDI from the verified result.'}
            {m.timing ? ` · plan ${m.timing.plan_s}s · compute ${m.timing.execute_s}s` : ''}</div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
