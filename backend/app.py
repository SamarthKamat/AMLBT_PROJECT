"""UDI 2.0 Flask API.

Flow for every question (the chat "agent pipeline"):
  Planner agent (Gemini or the local intent model) -> structured plan
  -> Validation gate (columns, types, filters, data sufficiency)          [never substitutes a column]
  -> Analyst / ML agent: one controlled Python operation -> verified result
  -> Business Analyst agent: findings + suggested actions from the verified numbers
  -> Narrator: Gemini wording accepted only if every number traces to the result (else the Python summary)
"""
from __future__ import annotations
import io
import json
import logging
import time
import traceback
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, Response, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
from config import DOMAINS, REAL_DATASETS
from artifacts import get_quality, get_relationships, get_semantics
from business_analyst import business_insights
from data_profiler import load_dataset
from data_quality import ALL_ACTIONS, DEFAULT_ACTIONS, assess_quality, clean_dataset, compare_quality, public
from errors import UDIError
from executor import execute
from explainer import digest
from gemini_client import GeminiService
from ops import Auth, JobQueue, Metrics, RateLimiter
from planner import PlanningService, explain_result
from reference_models import list_reference_models
from result_formatter import jsonsafe, records
from schemas import OPERATIONS, AnalysisPlan
from semantics import dimension_columns, kpi_columns, outcome_columns
from session_manager import Session, SessionStore
from storage import Storage
from validator import ClarificationNeeded, validate_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("udi.app")


def suggestions(s: Session) -> List[str]:
    """Example questions built from the uploaded schema (column semantics), so they always make sense for this data."""
    if s.df is None:
        return []
    sem = s.ctx.semantics
    kpis = kpi_columns(sem, "general", 3)
    outs = outcome_columns(sem)
    dims = dimension_columns(sem, s.profile, "general", 40, 4)
    geo = next((d for d in dims if sem[d]["semantic"] == "geography"), dims[0] if dims else None)
    tc = s.profile.get("primary_time_column")
    out = []
    if kpis and geo:
        out.append(f"Which {geo} has the highest {kpis[0]}?")
    if kpis:
        out.append(f"What is the average {kpis[-1]}?")
    out.append("How are the columns related?")
    if outs:
        out.append(f"What predicts {outs[0]}?")
        if dims:
            d2 = next((d for d in dims if d != outs[0]), None)
            if d2:
                out.append(f"{outs[0]} rate by {d2}")
    if kpis:
        out.append(f"Are there any outliers in {kpis[0]}?")
    if tc and kpis:
        out.append(f"Is {kpis[0]} increasing over time?")
        out.append(f"Forecast {kpis[0]} for the next 6 months")
        out.append(f"Why did {kpis[0]} change last quarter?")
    if len(dims) >= 2 and kpis:
        out.append(f"Total {kpis[0]} by {dims[0]} and {dims[1]}")
    out += ["Perform EDA", "Check data quality", "Segment the records"]
    fn = "hr" if any(w in " ".join(s.df.columns).lower() for w in ("attrition", "employee", "salary", "jobrole")) else \
         "marketing" if any(w in " ".join(s.df.columns).lower() for w in ("churn", "campaign", "contact")) else "sales"
    out.append(f"Generate a {fn} report")
    return list(dict.fromkeys(out))[:12]


def create_app(llm: Any = None, store: Optional[SessionStore] = None, planner_mode: Optional[str] = None,
               auth_enabled: Optional[bool] = None, data_dir: Optional[Path] = None) -> Flask:
    """`llm` must provide status(live), plan(...) and explain(...). It defaults to the real GeminiService."""
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app, origins=config.CORS_ORIGINS)
    app.json.sort_keys = False
    llm = llm or GeminiService()
    planner = PlanningService(llm, planner_mode)
    storage = Storage(Path(data_dir or config.DATA_DIR)) if config.PERSIST_SESSIONS or data_dir else None
    sessions = store or SessionStore(storage=storage)
    if store is not None and store.storage is None:
        storage = None                                   # an injected in-memory store stays in memory
    auth = Auth(storage or Storage(Path(data_dir or config.DATA_DIR)), config.AUTH_ENABLED if auth_enabled is None else auth_enabled)
    limiter = RateLimiter(config.RATE_LIMITS)
    jobs = JobQueue(config.JOB_WORKERS)
    metrics = Metrics()
    app.extensions.update(udi_llm=llm, udi_sessions=sessions, udi_planner=planner, udi_metrics=metrics, udi_jobs=jobs)

    # ------------------------------------------------------------------ plumbing
    @app.before_request
    def _start():
        g.t0 = time.perf_counter()
        g.request_id = uuid.uuid4().hex[:12]

    @app.after_request
    def _finish(resp):
        if request.path.startswith("/api/"):
            metrics.request(request.endpoint or request.path, time.perf_counter() - g.get("t0", time.perf_counter()), resp.status_code)
            resp.headers["X-Request-ID"] = g.get("request_id", "")
        return resp

    def user() -> Dict[str, Any]:
        if "user" not in g:
            g.user = auth.user_for(request.headers.get("Authorization"))
        return g.user

    def client_key() -> str:
        try:
            return user()["username"]
        except UDIError:
            return request.remote_addr or "anon"

    def own_session(sid: Optional[str], required: bool = True) -> Optional[Session]:
        s = sessions.get(sid)
        if s is None:
            if required:
                raise UDIError("SESSION_NOT_FOUND", "This session has expired or does not exist. Upload your dataset again to continue.")
            return None
        u = user()
        if u["role"] != "admin" and s.owner != u["username"]:
            raise UDIError("FORBIDDEN", "This session belongs to another user.")
        return s

    def need_data(s: Session) -> None:
        if s.df is None:
            raise UDIError("NO_DATASET", "Upload a dataset first.")

    # ------------------------------------------------------------------ errors
    @app.errorhandler(UDIError)
    def _udi_error(e: UDIError):
        metrics.error_code(e.code)
        body = e.to_dict()
        if getattr(e, "plan", None):
            body["plan"] = e.plan
        return jsonify(jsonsafe(body)), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_e):
        err = UDIError("FILE_TOO_LARGE", f"The file is larger than the {config.MAX_UPLOAD_MB} MB upload limit. Reduce the file (e.g. fewer rows/columns) and retry.")
        return jsonify(err.to_dict()), err.status

    @app.errorhandler(Exception)
    def _unexpected(e: Exception):
        if isinstance(e, HTTPException):
            if request.path.startswith("/api/"):
                return jsonify({"code": "BAD_REQUEST", "error": e.description, "retryable": False}), e.code
            return e
        log.error("Unhandled error [%s]: %s", g.get("request_id"), traceback.format_exc())
        err = UDIError("BACKEND_ERROR", f"Unexpected backend error: {type(e).__name__}: {str(e)[:200]}")
        return jsonify(err.to_dict()), err.status

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        live = request.args.get("check") in ("1", "true", "yes")
        gstat = llm.status(live=live)
        effective = "local" if planner.mode == "local" or (planner.mode == "auto" and not gstat.get("ready")) else "gemini"
        return jsonify({"ok": True, "version": config.VERSION, "gemini": gstat, "gemini_enabled": bool(gstat.get("ready")), "model": gstat.get("model"),
                        "planner": {"mode": planner.mode, "effective": effective, "local": planner.local.status()},
                        "auth_required": auth.enabled, "persistence": storage is not None,
                        "operations": OPERATIONS, "domains": list(DOMAINS), "status": "ready"})

    @app.get("/api/domains")
    def domains():
        return jsonify([{"id": k, "label": v["label"], "icon": v["icon"], "description": v["description"]} for k, v in DOMAINS.items()])

    @app.get("/api/reference-models")
    def reference_models():
        return jsonify(jsonsafe(list_reference_models()))

    @app.get("/api/metrics")
    def metrics_view():
        if user()["role"] != "admin":
            raise UDIError("FORBIDDEN", "Only administrators can read metrics.")
        return jsonify(jsonsafe(metrics.snapshot(sessions.active())))

    # ------------------------------------------------------------------ auth
    @app.post("/api/auth/register")
    def register():
        limiter.check(request.remote_addr or "anon", "auth")
        b = request.get_json(silent=True) or {}
        return jsonify(auth.register(b.get("username"), b.get("password")))

    @app.post("/api/auth/login")
    def login():
        limiter.check(request.remote_addr or "anon", "auth")
        b = request.get_json(silent=True) or {}
        return jsonify(auth.login(b.get("username"), b.get("password")))

    @app.post("/api/auth/logout")
    def logout():
        auth.logout((request.headers.get("Authorization") or "").removeprefix("Bearer ").strip())
        return jsonify({"ok": True})

    @app.get("/api/auth/me")
    def me():
        return jsonify({"user": user(), "auth_required": auth.enabled})

    # ------------------------------------------------------------------ sessions
    @app.post("/api/session")
    def session_create():
        data = request.get_json(silent=True) or {}
        domain = data.get("domain")
        if domain is not None and domain not in DOMAINS:
            raise UDIError("BAD_REQUEST", f"Unknown domain '{domain}'. Choose one of: {', '.join(DOMAINS)}.")
        s = sessions.create(domain, owner=user()["username"])
        return jsonify({"session_id": s.id})

    @app.get("/api/sessions")
    def session_list():
        u = user()
        if storage is None:
            return jsonify([])
        rows = storage.sessions_for(None if u["role"] == "admin" and request.args.get("all") else u["username"])
        return jsonify(jsonsafe([r for r in rows if r.get("dataset_name")]))

    @app.get("/api/session/<sid>")
    def session_state(sid):
        s = own_session(sid)
        out: Dict[str, Any] = {"session_id": s.id, "domain": s.domain, "has_dataset": s.df is not None}
        if s.df is not None:
            out.update(profile=s.profile, preview=records(s.df, 10), semantics=s.ctx.semantics, suggestions=suggestions(s),
                       cleaned=bool(s.cleaning_log) or (s.original_df is not None and len(s.original_df) != len(s.df)),
                       turns=[{"question": t["question"], "status": t["status"]} for t in s.turns])
        return jsonify(jsonsafe(out))

    @app.delete("/api/session/<sid>")
    def session_delete(sid):
        own_session(sid)
        sessions.delete(sid)
        return jsonify({"deleted": True, "session_id": sid})

    @app.post("/api/new-chat")
    def new_chat():
        data = request.get_json(silent=True) or {}
        domain = data.get("domain")
        if domain is not None and domain not in DOMAINS:
            raise UDIError("BAD_REQUEST", f"Unknown domain '{domain}'.")
        old = sessions.get(data.get("session_id"))
        if old is not None:
            own_session(old.id)
            sessions.delete(old.id)        # dataset, conversation context, cached models and stored files are discarded
        s = sessions.create(domain, owner=user()["username"])
        return jsonify({"session_id": s.id, "reset": True})

    @app.get("/api/session/<sid>/routing")
    def routing(sid):
        s = own_session(sid)
        return jsonify(jsonsafe({"session_id": sid, "routing": s.routing_log, "model_cache": {"entries": len(s.ctx.cache) if s.ctx else 0,
                                                                                               "hits": s.ctx.cache.hits if s.ctx else 0}}))

    @app.get("/api/session/<sid>/history")
    def history(sid):
        s = own_session(sid)
        rows = storage.history(sid) if storage else [{"operation": r.get("operation"), "title": r.get("question"), "planner": r.get("planner")} for r in s.routing_log]
        return jsonify(jsonsafe({"session_id": sid, "history": rows, "routing": s.routing_log}))

    # ------------------------------------------------------------------ datasets
    def _ingest(s: Session, df: pd.DataFrame, name: str, domain: str, content: bytes) -> Dict[str, Any]:
        if df.empty:
            raise UDIError("DATASET_PARSE_ERROR", "The uploaded dataset is empty.")
        import hashlib
        with s.lock:
            s.domain = domain
            s.cleaning_log = []
            s.original_df = df
            s.set_dataset(df, name, original=df, content_id=hashlib.sha1(content).hexdigest()[:16])
            sessions.persist_dataset(s)
        q = get_quality(s.ctx, df)
        log.info("UPLOAD session=%s domain=%s file=%s rows=%d cols=%d", s.id[:8], domain, name, len(df), df.shape[1])
        return jsonsafe({"session_id": s.id, "domain": domain, "profile": s.profile, "preview": records(df, 10), "semantics": s.ctx.semantics,
                         "quality": {"score": q["score"], "grade": q["grade"], "summary": q["summary"], "issues": len(q["issues"])},
                         "suggestions": suggestions(s)})

    @app.post("/api/upload")
    def upload():
        limiter.check(client_key(), "upload")
        domain = request.form.get("domain")
        if domain not in DOMAINS:
            raise UDIError("BAD_REQUEST", f"Select a valid domain ({', '.join(DOMAINS)}).")
        file = request.files.get("file")
        if not file or not file.filename:
            raise UDIError("BAD_REQUEST", "No dataset file was supplied.")
        ext = Path(file.filename).suffix.lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            raise UDIError("DATASET_PARSE_ERROR", "Only CSV, XLSX and XLS files are supported.")
        s = own_session(request.form.get("session_id"), required=False) or sessions.create(domain, owner=user()["username"])
        content = file.read()
        path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        try:
            df = load_dataset(path)
        finally:
            path.unlink(missing_ok=True)       # the raw upload never stays on disk
        return jsonify(_ingest(s, df, Path(file.filename).name, domain, content))

    @app.get("/api/samples")
    def samples():
        out = [{"id": f"sample:{d}", "label": f"{v['label']} (generated reference data)", "domain": d, "kind": "generated",
                "source": "UDI sample-data generator (seed 42)"} for d, v in DOMAINS.items() if (config.SAMPLE_DIR / f"{d}.csv").exists()]
        out += [{"id": f"real:{k}", "label": v["label"], "domain": v["domain"], "kind": "real", "source": v["source"], "use": v["use"]}
                for k, v in REAL_DATASETS.items() if (config.REAL_SAMPLE_DIR / v["file"]).exists()]
        return jsonify(out)

    @app.post("/api/samples/load")
    def sample_load():
        limiter.check(client_key(), "upload")
        b = request.get_json(silent=True) or {}
        sid, key = b.get("session_id"), str(b.get("id", ""))
        kind, _, name = key.partition(":")
        if kind == "sample" and name in DOMAINS:
            path, domain = config.SAMPLE_DIR / f"{name}.csv", name
        elif kind == "real" and name in REAL_DATASETS:
            path, domain = config.REAL_SAMPLE_DIR / REAL_DATASETS[name]["file"], b.get("domain") or REAL_DATASETS[name]["domain"]
        else:
            raise UDIError("BAD_REQUEST", "Unknown sample dataset.")
        if domain not in DOMAINS:
            domain = "company"
        s = own_session(sid, required=False) or sessions.create(domain, owner=user()["username"])
        df = load_dataset(path)
        return jsonify(_ingest(s, df, path.name, domain, path.read_bytes()))

    @app.get("/api/session/<sid>/download")
    def download_data(sid):
        s = own_session(sid)
        need_data(s)
        buf = io.BytesIO(s.df.to_csv(index=False).encode("utf-8-sig"))
        stem = Path(s.dataset_name or "dataset").stem
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"{stem}{'_cleaned' if s.cleaning_log else ''}.csv")

    # ------------------------------------------------------------------ data intelligence views
    @app.get("/api/session/<sid>/dictionary")
    def dictionary(sid):
        s = own_session(sid)
        need_data(s)
        rel = get_relationships(s.ctx, s.df)
        from eda_engine import heatmap_chart
        return jsonify(jsonsafe({"dictionary": rel["dictionary"], "findings": rel["findings"], "pairs": rel["association"]["pairs"][:60],
                                 "heatmap": heatmap_chart(rel["association"], "Column relationship heatmap", limit=18) if len(rel["association"]["columns"]) >= 2 else None,
                                 "hierarchies": rel["hierarchies"], "derived": rel["derived"], "date_rules": rel["date_rules"], "duplicates": rel["duplicates"],
                                 "candidate_keys": rel["candidate_keys"], "semantics": s.ctx.semantics}))

    def _run_direct(s: Session, plan: Dict[str, Any]) -> Dict[str, Any]:
        vp = validate_plan(AnalysisPlan(**plan), s.df, s.profile, s.roles)
        return execute(vp, s.ctx)

    @app.get("/api/session/<sid>/eda")
    def eda(sid):
        s = own_session(sid)
        need_data(s)
        col = request.args.get("column")
        with s.lock:
            return jsonify(jsonsafe(_run_direct(s, {"operation": "run_eda", "target_column": col} if col else {"operation": "run_eda"})))

    @app.get("/api/session/<sid>/quality")
    def quality(sid):
        s = own_session(sid)
        need_data(s)
        q = public(get_quality(s.ctx, s.df))
        return jsonify(jsonsafe({**q, "actions": {"default": DEFAULT_ACTIONS, "all": ALL_ACTIONS}, "cleaning_log": s.cleaning_log,
                                 "cleaned": bool(s.cleaning_log)}))

    @app.post("/api/session/<sid>/clean")
    def clean(sid):
        s = own_session(sid)
        need_data(s)
        b = request.get_json(silent=True) or {}
        actions = b.get("actions") or DEFAULT_ACTIONS
        with s.lock:
            before_df = s.df
            before = get_quality(s.ctx, before_df)
            try:
                cleaned, logrows = clean_dataset(before_df, s.profile, s.ctx.semantics, actions)
            except ValueError as e:
                raise UDIError("BAD_REQUEST", str(e))
            original = s.original_df if s.original_df is not None else before_df
            s.set_dataset(cleaned, s.dataset_name or "dataset", original=original)
            s.cleaning_log = (s.cleaning_log or []) + logrows
            after = get_quality(s.ctx, cleaned)
            sessions.persist_dataset(s, cleaned=True)
            cmp = compare_quality(before, after, before_df, cleaned)
        return jsonify(jsonsafe({"log": logrows, "comparison": cmp, "profile": s.profile, "preview": records(cleaned, 10), "quality": public(after),
                                 "suggestions": suggestions(s)}))

    @app.post("/api/session/<sid>/revert")
    def revert(sid):
        s = own_session(sid)
        need_data(s)
        if s.original_df is None:
            raise UDIError("BAD_REQUEST", "There is no original version to revert to.")
        with s.lock:
            s.set_dataset(s.original_df, s.dataset_name or "dataset", original=s.original_df)
            s.cleaning_log = []
            sessions.persist_dataset(s)
        return jsonify(jsonsafe({"reverted": True, "profile": s.profile, "preview": records(s.df, 10), "quality": public(get_quality(s.ctx, s.df))}))

    @app.get("/api/session/<sid>/suggestions")
    def suggest(sid):
        s = own_session(sid)
        return jsonify(suggestions(s))

    # ------------------------------------------------------------------ reports (background jobs)
    @app.post("/api/session/<sid>/report")
    def report_start(sid):
        limiter.check(client_key(), "report")
        s = own_session(sid)
        need_data(s)
        b = request.get_json(silent=True) or {}
        fn = b.get("function") or "general"
        target = b.get("target") or None
        if target and target not in s.df.columns:
            raise UDIError("INVALID_COLUMN", f"'{target}' is not a column of the dataset.", details={"available": list(s.df.columns)})
        from agents import build_report

        def work(progress):
            with s.lock:
                return build_report(s.df, s.ctx, fn, s.dataset_name or "dataset", progress=progress, target=target)
        jid = jobs.submit(user()["username"], "report", work)
        return jsonify({"job_id": jid, "status": "queued"})

    def _job(jid: str) -> Dict[str, Any]:
        j = jobs.get(jid)
        if not j:
            raise UDIError("JOB_NOT_FOUND", "This job does not exist or has expired.")
        u = user()
        if u["role"] != "admin" and j["owner"] != u["username"]:
            raise UDIError("FORBIDDEN", "This job belongs to another user.")
        return j

    @app.get("/api/jobs/<jid>")
    def job_status(jid):
        j = _job(jid)
        out = {k: j[k] for k in ("id", "kind", "status", "progress", "message", "error")}
        if j["status"] == "done":
            out["result"] = j["result"]
        return jsonify(jsonsafe(out))

    @app.get("/api/jobs/<jid>/download")
    def job_download(jid):
        j = _job(jid)
        if j["status"] != "done":
            raise UDIError("BAD_REQUEST", "The report is not ready yet.")
        rep = j["result"]
        fmt = request.args.get("format", "pdf")
        stem = f"UDI_{rep['function']}_report"
        from report_engine import report_pdf, report_xlsx
        if fmt == "pdf":
            return send_file(io.BytesIO(report_pdf(rep)), mimetype="application/pdf", as_attachment=True, download_name=f"{stem}.pdf")
        if fmt == "xlsx":
            return send_file(io.BytesIO(report_xlsx(rep)), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name=f"{stem}.xlsx")
        if fmt == "json":
            return Response(json.dumps(rep, default=str, indent=2), mimetype="application/json", headers={"Content-Disposition": f"attachment; filename={stem}.json"})
        raise UDIError("BAD_REQUEST", "format must be pdf, xlsx or json.")

    # ------------------------------------------------------------------ chat
    @app.post("/api/chat")
    def chat():
        limiter.check(client_key(), "chat")
        body = request.get_json(silent=True) or {}
        s = own_session(body.get("session_id"))
        need_data(s)
        question = (body.get("message") or "").strip()
        if not question:
            raise UDIError("BAD_REQUEST", "Please enter a question.")
        if len(question) > 2000:
            raise UDIError("BAD_REQUEST", "The question is too long (2,000 characters at most).")
        with s.lock:
            return jsonify(jsonsafe(_handle_question(s, question)))

    def _record_turn(s: Session, turn: Dict[str, Any], planner_src: Optional[str]) -> None:
        s.turns.append(turn)
        if storage:
            try:
                storage.add_turn(s.id, turn, planner_src)
            except Exception as exc:                    # persistence must never break an answer
                log.warning("Could not persist turn: %s", exc)

    def _handle_question(s: Session, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        route: Dict[str, Any] = {"question": question, "at": time.strftime("%H:%M:%S")}
        plan, src, pnote, pinfo = None, None, None, {}
        trace: List[Dict[str, Any]] = []
        try:
            plan, src, pnote, pinfo = planner.plan(question, s)
            t_plan = time.perf_counter()
            trace.append({"agent": "Planner agent", "tool": "gemini" if src == "gemini" else "local intent model", "status": "ok", "seconds": round(t_plan - t0, 2),
                          "headline": f"{plan.operation}" + (f" on {plan.target_column}" if plan.target_column else "") + (f" by {', '.join(plan.group_by)}" if plan.group_by else "")})
            route.update(operation=plan.operation, planned_target=plan.target_column, planner=src)
            try:
                vp = validate_plan(plan, s.df, s.profile, s.roles)
            except ClarificationNeeded as c:
                opts = c.options or pinfo.get("options") or []
                s.pending_clarification = {"asked": c.question, "original_question": question, "options": opts[:15]}
                _record_turn(s, {"question": question, "status": "clarification", "plan": plan.model_dump(exclude_none=True), "clarification": c.question}, src)
                route.update(status="clarification")
                s.routing_log.append(route)
                trace.append({"agent": "Validation gate", "tool": "validate_plan", "status": "clarify", "seconds": 0.0, "headline": c.question})
                return {"type": "clarification", "answer": c.question, "options": opts, "plan": plan.model_dump(exclude_none=True), "result": None,
                        "answer_source": "udi", "planner": src, "planner_note": pnote, "agent_trace": trace, "timing": {"plan_s": round(t_plan - t0, 2)}}
            trace.append({"agent": "Validation gate", "tool": "validate_plan", "status": "ok", "seconds": round(time.perf_counter() - t_plan, 2),
                          "headline": f"{len(vp.data):,} rows; columns verified" + (f"; filters: {'; '.join(vp.filters_applied)}" if vp.filters_applied else "")})
            s.pending_clarification = None
            executed = vp.plan.model_dump(exclude_none=True)
            t_v = time.perf_counter()
            result = execute(vp, s.ctx)
            t_exec = time.perf_counter()
            agent = "ML agent" if result["provenance"]["uses_machine_learning"] else "Data Analyst agent"
            trace.append({"agent": agent, "tool": vp.plan.operation, "status": "ok", "seconds": round(t_exec - t_v, 2), "headline": result["summary"][:200]})
            result["business_insights"] = business_insights(result, executed, vp.data, s.ctx.semantics)
            if vp.plan.chart_type:
                result["preferred_chart"] = vp.plan.chart_type
            trace.append({"agent": "Business Analyst agent", "tool": "business_insights", "status": "ok", "seconds": 0.0,
                          "headline": f"{len(result['business_insights'])} finding(s)"})
            use_llm = src == "gemini"
            answer, source, note = explain_result(llm, question, vp.plan, result, use_llm=use_llm)
            t_end = time.perf_counter()
            trace.append({"agent": "Narrator", "tool": "gemini (grounded)" if source == "gemini" else "python summary", "status": "ok", "seconds": round(t_end - t_exec, 2),
                          "headline": "every number checked against the verified result" if source == "gemini" else "verified Python summary"})
            model = (result.get("evaluation") or {}).get("selected_model")
            route.update(status="ok", operation=vp.plan.operation, target=vp.plan.target_column, group_by=vp.plan.group_by,
                         time_column=vp.time_column if vp.plan.operation in ("get_current_state", "analyze_trend", "forecast", "explain_change") else None,
                         aggregation=vp.plan.aggregation, filters=vp.filters_applied, model=model, ml=result["provenance"]["uses_machine_learning"],
                         seconds=round(t_end - t0, 2))
            s.routing_log.append(route)
            metrics.answered(vp.plan.operation, src)
            log.info("ROUTE session=%s planner=%s q=%r -> op=%s target=%s group_by=%s model=%s", s.id[:8], src, question, vp.plan.operation,
                     vp.plan.target_column, vp.plan.group_by, model)
            _record_turn(s, {"question": question, "status": "ok", "plan": executed, "digest": digest(result)}, src)
            if storage:
                try:
                    storage.add_history(s.id, vp.plan.operation, result.get("title") or "", result.get("summary") or "", round(t_end - t0, 2), src)
                except Exception:
                    pass
            notes = [n for n in (pnote, note) if n]
            return {"type": "answer", "answer": answer, "answer_source": source, "explanation_note": " ".join(notes) or None, "plan": executed, "result": result,
                    "planner": src, "planner_info": pinfo or None, "agent_trace": trace,
                    "timing": {"plan_s": round(t_plan - t0, 2), "execute_s": round(t_exec - t_v, 2), "explain_s": round(t_end - t_exec, 2)}}
        except UDIError as e:
            route.update(status="error", error=e.code, seconds=round(time.perf_counter() - t0, 2))
            s.routing_log.append(route)
            _record_turn(s, {"question": question, "status": "error", "plan": plan.model_dump(exclude_none=True) if plan else {}, "error": f"{e.code}: {e.message[:200]}"}, src)
            log.info("ROUTE session=%s q=%r -> ERROR %s", s.id[:8], question, e.code)
            e.plan = plan.model_dump(exclude_none=True) if plan else None
            raise

    # ------------------------------------------------------------------ frontend
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def frontend(path):
        if path.startswith("api/"):
            raise UDIError("BAD_REQUEST", "Unknown API endpoint.", status=404)
        dist = config.FRONTEND_DIST
        if path and (dist / path).is_file():
            return send_from_directory(dist, path)
        if (dist / "index.html").exists():
            return send_from_directory(dist, "index.html")
        return jsonify({"message": "UDI API is running. Build the frontend with `npm run build` in /frontend (or use `npm run dev`)."})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=False, threaded=True)
