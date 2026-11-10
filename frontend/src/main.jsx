import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Upload, Send, Plus, Sparkles, Brain, ShieldCheck, ChevronDown, BarChart3, FileSpreadsheet, Bot, User, RefreshCw, AlertOctagon, HelpCircle,
  MessageSquare, Network, FileText, History, Database, Trash2, LogOut, Cpu, Wand2,
} from 'lucide-react';
import ResultView, { AgentTrace } from './ResultView';
import DictionaryTab from './tabs/DictionaryTab';
import EdaTab from './tabs/EdaTab';
import QualityTab from './tabs/QualityTab';
import ReportsTab from './tabs/ReportsTab';
import HistoryTab from './tabs/HistoryTab';
import { apiFetch, getJson, postJson, del, ApiError, ERROR_INFO, getToken, setToken } from './api';
import { fmt } from './format';
import './styles.css';
import './extra.css';
import './ui2.css';

const fallbackDomains = [
  { id: 'company', label: 'Company', icon: '▣', description: 'Sales, demand, inventory, customers and profitability.' },
  { id: 'hospital', label: 'Hospital', icon: '✚', description: 'Patient load, capacity, waiting time and staffing.' },
  { id: 'college', label: 'College', icon: '⌂', description: 'Attendance, students, faculty and resource utilization.' },
  { id: 'municipality', label: 'Municipality', icon: '◈', description: 'Waste, complaints, service demand and zones.' },
  { id: 'ngo', label: 'NGO', icon: '♡', description: 'Beneficiaries, donations, aid and volunteer demand.' },
  { id: 'community', label: 'Community', icon: '⌁', description: 'Local requests, participation, shortages and needs.' },
  { id: 'event_organizer', label: 'Event Organizer', icon: '★', description: 'Attendance, capacity, registrations and resources.' },
];
const TABS = [
  ['chat', 'Ask', MessageSquare], ['dictionary', 'Relationships', Network], ['eda', 'EDA', BarChart3],
  ['quality', 'Data quality', ShieldCheck], ['reports', 'Reports', FileText], ['history', 'History', History],
];
const SESSION_KEY = 'udi.session';
const remember = (sid) => { try { sid ? localStorage.setItem(SESSION_KEY, sid) : localStorage.removeItem(SESSION_KEY); } catch { /* ignore */ } };
const recall = () => { try { return localStorage.getItem(SESSION_KEY) || ''; } catch { return ''; } };

function statusView(health) {
  if (!health) return { cls: '', text: 'Checking backend…', title: '' };
  if (health.backend_offline) return { cls: 'bad', text: 'Backend offline', title: 'Start the backend: python serve.py (in /backend)' };
  const g = health.gemini || {};
  const p = health.planner || {};
  if (p.effective === 'local') return { cls: 'live', text: 'Local planner (offline)', title: p.local?.detail + (p.mode === 'auto' ? ' Add GEMINI_API_KEY to backend/.env to plan with Gemini.' : '') };
  if (g.status === 'ok') return { cls: 'live', text: `Gemini ready · ${g.model}`, title: g.detail };
  if (g.status === 'configured') return { cls: 'live', text: `Gemini configured · ${g.model}`, title: 'Key and model are set. Click to verify the connection.' };
  if (g.status === 'model_unavailable') return { cls: 'bad', text: `Gemini model unavailable · ${g.model}`, title: `${g.detail}${g.available_models ? ' Available: ' + g.available_models.join(', ') : ''}` };
  return { cls: 'bad', text: 'Gemini unreachable', title: g.detail };
}

function AuthScreen({ onDone }) {
  const [mode, setMode] = useState('login');
  const [u, setU] = useState('');
  const [p, setP] = useState('');
  const [err, setErr] = useState(null);
  const submit = async (e) => {
    e.preventDefault(); setErr(null);
    try { const r = await postJson(`/api/auth/${mode}`, { username: u, password: p }); setToken(r.token); onDone(r.user); } catch (x) { setErr(x); }
  };
  return (
    <div className="authWrap">
      <form className="authCard" onSubmit={submit}>
        <div className="brandMark"><Sparkles size={20} /></div>
        <h2>UDI 2.0</h2><p className="muted">{mode === 'login' ? 'Sign in to your workspace' : 'Create an account (the first account becomes the administrator)'}</p>
        <input placeholder="username" value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" />
        <input placeholder="password (8+ characters)" type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} />
        {err && <div className="errorCard"><p>{err.message}</p></div>}
        <button className="primary" type="submit">{mode === 'login' ? 'Sign in' : 'Create account'}</button>
        <button type="button" className="linkBtn" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>{mode === 'login' ? 'No account? Register' : 'Have an account? Sign in'}</button>
      </form>
    </div>
  );
}

function App() {
  const [domains, setDomains] = useState(fallbackDomains);
  const [domain, setDomain] = useState('company');
  const [session, setSession] = useState('');
  const [profile, setProfile] = useState(null);
  const [quality, setQuality] = useState(null);
  const [suggest, setSuggest] = useState([]);
  const [samples, setSamples] = useState([]);
  const [showCols, setShowCols] = useState(false);
  const [showDomains, setShowDomains] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [health, setHealth] = useState(null);
  const [tab, setTab] = useState('chat');
  const [version, setVersion] = useState(0);
  const [me, setMe] = useState(null);
  const [needAuth, setNeedAuth] = useState(false);
  const fileRef = useRef(null);
  const endRef = useRef(null);

  const checkHealth = useCallback(async (live = false) => {
    try { const h = await apiFetch('/api/health' + (live ? '?check=1' : ''), {}, 30000); setHealth(h); return h; } catch { setHealth({ backend_offline: true }); return null; }
  }, []);

  const applyDataset = (data, name) => {
    setSession(data.session_id); remember(data.session_id);
    setProfile(data.profile); setSuggest(data.suggestions || []); setQuality(data.quality || null);
    setVersion((v) => v + 1);
    const p = data.profile;
    setMessages([{ role: 'assistant', content: `${name || p.dataset_name} is loaded: ${fmt(p.rows)} rows, ${p.columns.length} columns (${p.numeric_columns.length} numeric, ${p.categorical_columns.length} categorical, ${p.datetime_columns.length} date/time).${p.date_range ? ` Period covered: ${String(p.date_range.start).slice(0, 10)} to ${String(p.date_range.end).slice(0, 10)}.` : ''}${data.quality ? ` Data quality ${fmt(data.quality.score, 1)}/100 (grade ${data.quality.grade}).` : ''} Ask anything — or open the Relationships, EDA, Data quality and Reports tabs.`, warnings: p.warnings }]);
  };

  const boot = useCallback(async () => {
    const h = await checkHealth(false);
    if (h?.auth_required && !getToken()) { setNeedAuth(true); return; }
    try { const m = await getJson('/api/auth/me'); setMe(m.user); } catch (e) { if (e.code === 'UNAUTHORIZED') { setToken(''); setNeedAuth(true); return; } }
    apiFetch('/api/domains').then((d) => Array.isArray(d) && setDomains(d)).catch(() => {});
    getJson('/api/samples').then(setSamples).catch(() => {});
    const saved = recall();
    if (saved) {
      try {
        const st = await getJson(`/api/session/${saved}`);
        setSession(saved); setDomain(st.domain || 'company');
        if (st.has_dataset) { applyDataset({ session_id: saved, profile: st.profile, suggestions: st.suggestions }, st.profile.dataset_name); return; }
        return;
      } catch { remember(''); }
    }
    postJson('/api/session', { domain: 'company' }).then((d) => { if (d?.session_id) { setSession(d.session_id); remember(d.session_id); } }).catch(() => {});
  }, [checkHealth]);

  useEffect(() => { boot(); }, [boot]);
  useEffect(() => { if (tab === 'chat') endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, tab]);

  const current = domains.find((d) => d.id === domain) || fallbackDomains[0];
  const st = statusView(health);

  const resetSession = async (nextDomain) => {
    try { const data = await postJson('/api/new-chat', { session_id: session, domain: nextDomain }); setSession(data.session_id); remember(data.session_id); }
    catch { try { const d = await postJson('/api/session', { domain: nextDomain }); setSession(d.session_id); remember(d.session_id); } catch { /* shown on next request */ } }
    setMessages([]); setProfile(null); setInput(''); setShowCols(false); setQuality(null); setSuggest([]); setTab('chat'); setVersion((v) => v + 1);
  };
  const selectDomain = async (id) => { setDomain(id); setShowDomains(false); await resetSession(id); };
  const newChat = async () => { await resetSession(domain); };
  const deleteData = async () => {
    if (!session || !window.confirm('Delete this dataset, its conversation and all stored copies from the server?')) return;
    try { await del(`/api/session/${session}`); } catch { /* already gone */ }
    remember(''); setSession(''); await resetSession(domain);
  };

  const upload = async (file) => {
    if (!file) return;
    setUploading(true);
    const fd = new FormData();
    fd.append('file', file); fd.append('domain', domain); fd.append('session_id', session);
    try { const data = await apiFetch('/api/upload', { method: 'POST', body: fd }); applyDataset(data, file.name); setTab('chat'); }
    catch (e) { setMessages([{ role: 'assistant', error: e }]); }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = ''; }
  };
  const loadSample = async (id) => {
    if (!id) return;
    setUploading(true);
    try { const data = await postJson('/api/samples/load', { id, session_id: session, domain }); applyDataset(data); setTab('chat'); }
    catch (e) { setMessages([{ role: 'assistant', error: e }]); }
    finally { setUploading(false); }
  };

  const ask = async (question) => {
    const query = (question ?? input).trim();
    if (!query || loading || !profile) return;
    setInput(''); setTab('chat');
    setMessages((prev) => [...prev, { role: 'user', content: query }, { role: 'assistant', loading: true }]);
    setLoading(true);
    try {
      const data = await postJson('/api/chat', { session_id: session, message: query });
      setMessages((prev) => [...prev.slice(0, -1), { role: 'assistant', content: data.answer, kind: data.type, options: data.options, source: data.answer_source, note: data.explanation_note,
        plan: data.plan, result: data.result, timing: data.timing, question: query, planner: data.planner, trace: data.agent_trace, plannerNote: data.planner_note }]);
    } catch (e) {
      setMessages((prev) => [...prev.slice(0, -1), { role: 'assistant', error: e, question: query }]);
    } finally { setLoading(false); }
  };

  const onCleaned = (r) => {
    setProfile(r.profile); if (r.quality) setQuality({ score: r.quality.score, grade: r.quality.grade });
    if (r.suggestions) setSuggest(r.suggestions);
    setMessages((m) => [...m, { role: 'assistant', content: r.reverted ? 'The original dataset was restored. Earlier answers refer to the cleaned version.' : `Cleaning applied: ${r.log.length} change(s). Quality score ${fmt(r.comparison.score_before, 1)} → ${fmt(r.comparison.score_after, 1)}. New questions use the cleaned data.` }]);
  };

  if (needAuth) return <AuthScreen onDone={(u) => { setMe(u); setNeedAuth(false); boot(); }} />;

  const colNames = profile ? profile.columns.map((c) => c.name) : [];
  return (
    <div className="app">
      <div className="orb orb1" /><div className="orb orb2" /><div className="orb orb3" />
      <header className="topbar">
        <div className="brand">
          <div className="brandMark"><Sparkles size={20} /></div>
          <div><div className="brandName">UDI 2.0 · Universal Dataset Intelligence</div>
            <div className="brandSub">Autonomous data analyst & business decision intelligence · every number computed in Python</div></div>
        </div>
        <div className="topRight">
          <button className="status" onClick={() => checkHealth(true)} title={st.title || 'Click to re-check'}><span className={'dot ' + st.cls} />{st.text}<RefreshCw size={12} className="refresh" /></button>
          {me && health?.auth_required && <button className="status" onClick={async () => { await postJson('/api/auth/logout', {}); setToken(''); setNeedAuth(true); }} title="Sign out"><User size={13} />&nbsp;{me.username}<LogOut size={12} className="refresh" /></button>}
        </div>
      </header>

      <main className="shell">
        {!profile && (
          <section className="hero">
            <div className="heroTag"><span>●</span>PLANNER INTERPRETS · PYTHON COMPUTES · AGENTS EXPLAIN</div>
            <h1>Ask your data <em>anything.</em><br />Get decisions, not just charts.</h1>
            <p>Upload a dataset (or load one of the bundled real-world datasets) and ask in plain language: “which country has the highest sales?”, “why did profit drop last quarter?”, “what predicts attrition?”. UDI validates every plan against your real columns, computes the answer in Python, explains models with SHAP, scores data quality and writes business reports for marketing, HR, operations, finance and sales.</p>
            <div className="heroPills">
              <span><ShieldCheck size={15} />Plans validated against your data</span>
              <span><Brain size={15} />AutoML + SHAP explanations</span>
              <span><Network size={15} />Column relationships mapped</span>
              <span><Wand2 size={15} />Quality score + auto-cleaning</span>
              <span><FileText size={15} />PDF / Excel business reports</span>
            </div>
          </section>
        )}

        <div className="layout">
          <aside className="sidebar">
            <div className="panel uploadPanel">
              <div className="panelTitle">Dataset</div>
              <button className="upload" onClick={() => fileRef.current?.click()} disabled={uploading}>
                <Upload size={18} /><span>{uploading ? 'Reading and profiling…' : 'Upload CSV / Excel'}</span>
              </button>
              <input ref={fileRef} type="file" hidden accept=".csv,.xlsx,.xls" onChange={(e) => upload(e.target.files?.[0])} />
              {samples.length > 0 && (
                <select className="select full" value="" onChange={(e) => loadSample(e.target.value)} disabled={uploading}>
                  <option value="">…or load a bundled dataset</option>
                  <optgroup label="Real public datasets">{samples.filter((s) => s.kind === 'real').map((s) => <option key={s.id} value={s.id}>{s.label} — {s.use}</option>)}</optgroup>
                  <optgroup label="Generated reference datasets">{samples.filter((s) => s.kind !== 'real').map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}</optgroup>
                </select>
              )}
              {profile && (
                <div className="profile">
                  <div className="profileHead"><FileSpreadsheet size={15} /><span>{profile.dataset_name}</span></div>
                  <div className="statGrid">
                    <div><b>{fmt(profile.rows)}</b><small>Rows</small></div>
                    <div><b>{profile.columns.length}</b><small>Columns</small></div>
                    <div><b>{fmt(profile.missing_total)}</b><small>Missing cells</small></div>
                    <div><b>{quality ? `${fmt(quality.score, 0)}` : fmt(profile.duplicate_rows)}</b><small>{quality ? `Quality (${quality.grade})` : 'Duplicate rows'}</small></div>
                  </div>
                  <button className="details slim" onClick={() => setShowCols(!showCols)}>Columns and detected types <ChevronDown size={14} className={showCols ? 'rot' : ''} /></button>
                  {showCols && <div className="colList">{profile.columns.map((c) => <div key={c.name} title={c.note || ''}><b>{c.name}</b><span className={'role ' + c.role}>{c.role}</span></div>)}</div>}
                  <button className="details slim danger" onClick={deleteData}><span><Trash2 size={13} /> Delete dataset from server</span></button>
                </div>
              )}
            </div>

            <div className="panel">
              <button className="panelTitle toggle" onClick={() => setShowDomains(!showDomains)}>Domain context: <b>{current.label}</b><ChevronDown size={14} className={showDomains ? 'rot' : ''} /></button>
              {showDomains && <div className="domainList">{domains.map((item) => (
                <button key={item.id} className={'domain ' + (domain === item.id ? 'active' : '')} onClick={() => selectDomain(item.id)}>
                  <span className="domainIcon">{item.icon}</span><span><b>{item.label}</b><small>{item.description}</small></span></button>))}</div>}
              <div className="tableNote">Context for the planner only — it never picks a model or a target.</div>
            </div>

            {profile && suggest.length > 0 && (
              <div className="panel">
                <div className="panelTitle">Try asking</div>
                <div className="suggestList">{suggest.map((s) => <button key={s} onClick={() => ask(s)} disabled={loading}>{s}</button>)}</div>
              </div>
            )}
            <button className="newChat" onClick={newChat}><Plus size={17} />New chat</button>
          </aside>

          <section className="chatArea">
            <div className="chatHeader">
              <div><span className="eyebrow">{current.label.toUpperCase()} WORKSPACE</span><h2>{profile ? profile.dataset_name : 'No dataset yet'}</h2></div>
              <div className="engine"><span className={'pulse ' + (loading ? 'busy' : '')} />{loading ? 'Planning and computing…' : profile ? 'Ready' : 'Waiting for a dataset'}</div>
            </div>
            {profile && (
              <nav className="tabs">{TABS.map(([id, name, Icon]) => <button key={id} className={tab === id ? 'on' : ''} onClick={() => setTab(id)}><Icon size={14} />{name}</button>)}</nav>
            )}

            {tab === 'chat' && (
              <>
                {!profile && messages.length === 0 && (
                  <div className="empty">
                    <div className="emptyIcon"><Database size={28} /></div>
                    <h3>Start with your dataset</h3>
                    <p>Upload a CSV or Excel file, or load a bundled real dataset (Global Superstore, IBM HR Attrition, IBM Telco Churn, UCI Bank Marketing). You do not choose a target or a model: each question is planned, checked against your columns and computed on demand.</p>
                    <div className="cards">
                      <div className="feature"><Brain /><b>Ask naturally</b><span>“Which country has the highest sales?” “What predicts churn?” “Why did profit fall?”</span></div>
                      <div className="feature"><BarChart3 /><b>Answers with visuals</b><span>Every answer comes with charts you can switch (bar, pie, line, table) and export.</span></div>
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
                  <div className="composerHint">Enter to send · Shift + Enter for a new line · follow-ups like “only for Goa” or “and by region?” reuse the previous question</div>
                </div>
              </>
            )}
            {profile && tab === 'dictionary' && <DictionaryTab session={session} version={version} />}
            {profile && tab === 'eda' && <EdaTab session={session} version={version} columns={colNames} />}
            {profile && tab === 'quality' && <QualityTab session={session} version={version} onChanged={onCleaned} />}
            {profile && tab === 'reports' && <ReportsTab session={session} version={version} columns={colNames} />}
            {profile && tab === 'history' && <HistoryTab session={session} version={version + messages.length} />}
          </section>
        </div>
        <footer className="foot">UDI {health?.version || '2.0'} · Goa Institute of Management · AMLBT project · planner: {health?.planner?.effective || '…'} · {health?.persistence ? 'sessions persisted' : 'in-memory sessions'}</footer>
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
    return <div className="msg assistant"><div className="avatar ai"><Bot size={16} /></div><div className="bubble typing"><span /><span /><span /></div></div>;
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
        {m.kind === 'clarification' && m.options?.length > 0 && <div className="chips">{m.options.map((o) => <button key={o} onClick={() => onAsk(o)} disabled={busy}>{o}</button>)}</div>}
        {m.note && <div className="sourceNote">{m.note}</div>}
        {m.warnings?.length > 0 && m.warnings.map((w, i) => <div className="sourceNote" key={i}>{w}</div>)}
        {m.result && <ResultView result={m.result} plan={m.plan} />}
        {m.trace && <AgentTrace steps={m.trace} title="How UDI answered (agent pipeline)" />}
        {m.result && m.source && (
          <div className="sourceTag"><Cpu size={11} /> Planned by {m.planner === 'gemini' ? 'Gemini' : 'the local intent model'} · {m.source === 'gemini' ? 'wording by Gemini; every number checked against the Python result' : 'wording written by UDI from the verified result'}
            {m.timing ? ` · plan ${m.timing.plan_s}s · compute ${m.timing.execute_s}s` : ''}</div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
