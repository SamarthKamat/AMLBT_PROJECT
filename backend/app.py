"""UDI Flask API.

Flow for every question:  profile + context -> Gemini plan -> validation against the real data
-> Python operation -> verified result -> explanation grounded in that result.
"""
from __future__ import annotations
import hashlib
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
from config import DOMAINS
from context import ExecContext
from data_profiler import load_dataset, profile_dataframe, profile_for_llm
from errors import UDIError
from executor import execute
from gemini_client import GeminiService
from planner import PlanningService, explain_result
from explainer import digest
from reference_models import list_reference_models
from result_formatter import jsonsafe, records
from schemas import OPERATIONS
from session_manager import Session, SessionStore
from validator import ClarificationNeeded, roles_from_profile, validate_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("udi.app")


def create_app(llm: Any = None, store: Optional[SessionStore] = None) -> Flask:
    """`llm` must provide status(live), plan(...) and explain(...). It defaults to the real GeminiService."""
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app)
    app.json.sort_keys = False          # keep table columns / metrics in the order Python produced them
    llm = llm or GeminiService()
    planner = PlanningService(llm)
    sessions = store or SessionStore()
    app.extensions["udi_llm"] = llm
    app.extensions["udi_sessions"] = sessions

    # ------------------------------------------------------------------ errors
    @app.errorhandler(UDIError)
    def _udi_error(e: UDIError):
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
        log.error("Unhandled error: %s", traceback.format_exc())
        err = UDIError("BACKEND_ERROR", f"Unexpected backend error: {type(e).__name__}: {str(e)[:200]}")
        return jsonify(err.to_dict()), err.status

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        live = request.args.get("check") in ("1", "true", "yes")
        g = llm.status(live=live)
        return jsonify({"ok": True, "gemini": g, "gemini_enabled": bool(g.get("ready")), "model": g.get("model"),
                        "operations": OPERATIONS, "domains": list(DOMAINS), "status": "ready" if g.get("ready") else "gemini_not_ready"})

    @app.get("/api/domains")
    def domains():
        return jsonify([{"id": k, "label": v["label"], "icon": v["icon"], "description": v["description"]} for k, v in DOMAINS.items()])

    @app.get("/api/reference-models")
    def reference_models():
        return jsonify(jsonsafe(list_reference_models()))

    # ------------------------------------------------------------------ sessions
    @app.post("/api/session")
    def session_create():
        data = request.get_json(silent=True) or {}
        domain = data.get("domain")
        if domain is not None and domain not in DOMAINS:
            raise UDIError("BAD_REQUEST", f"Unknown domain '{domain}'. Choose one of: {', '.join(DOMAINS)}.")
        s = sessions.create(domain)
        return jsonify({"session_id": s.id})

    @app.post("/api/new-chat")
    def new_chat():
        data = request.get_json(silent=True) or {}
        domain = data.get("domain")
        if domain is not None and domain not in DOMAINS:
            raise UDIError("BAD_REQUEST", f"Unknown domain '{domain}'.")
        sessions.delete(data.get("session_id"))       # dataset, conversation context and cached models are all discarded
        s = sessions.create(domain)
        return jsonify({"session_id": s.id, "reset": True})

    @app.get("/api/session/<sid>/routing")
    def routing(sid):
        s = sessions.get(sid)
        if not s:
            raise UDIError("SESSION_NOT_FOUND", "Session not found or expired.")
        return jsonify(jsonsafe({"session_id": sid, "routing": s.routing_log, "model_cache": {"entries": len(s.ctx.cache) if s.ctx else 0,
                                                                                               "hits": s.ctx.cache.hits if s.ctx else 0}}))

    # ------------------------------------------------------------------ upload
    @app.post("/api/upload")
    def upload():
        domain = request.form.get("domain")
        if domain not in DOMAINS:
            raise UDIError("BAD_REQUEST", f"Select a valid domain ({', '.join(DOMAINS)}).")
        file = request.files.get("file")
        if not file or not file.filename:
            raise UDIError("BAD_REQUEST", "No dataset file was supplied.")
        ext = Path(file.filename).suffix.lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            raise UDIError("DATASET_PARSE_ERROR", "Only CSV, XLSX and XLS files are supported.")
        s = sessions.get(request.form.get("session_id")) or sessions.create(domain)
        content = file.read()
        path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        try:
            df = load_dataset(path)
        finally:
            path.unlink(missing_ok=True)
        if df.empty:
            raise UDIError("DATASET_PARSE_ERROR", "The uploaded dataset is empty.")
        profile = profile_dataframe(df, file.filename)
        roles = roles_from_profile(profile)
        with s.lock:
            s.domain = domain
            s.df, s.profile, s.roles = df, profile, roles
            s.llm_profile = profile_for_llm(profile, df, config.GEMINI_SAMPLE_ROWS)
            s.ctx = ExecContext(roles=roles, profile=profile, dataset_id=hashlib.sha1(content).hexdigest()[:16])
            s.reset_conversation()
        log.info("UPLOAD session=%s domain=%s file=%s rows=%d cols=%d", s.id, domain, file.filename, len(df), df.shape[1])
        return jsonify(jsonsafe({"session_id": s.id, "domain": domain, "profile": profile, "preview": records(df, 10)}))

    # ------------------------------------------------------------------ chat
    @app.post("/api/chat")
    def chat():
        body = request.get_json(silent=True) or {}
        s = sessions.get(body.get("session_id"))
        if s is None:
            raise UDIError("SESSION_NOT_FOUND", "This session has expired or does not exist. Upload your dataset again to continue.")
        if s.df is None:
            raise UDIError("NO_DATASET", "Upload a dataset first.")
        question = (body.get("message") or "").strip()
        if not question:
            raise UDIError("BAD_REQUEST", "Please enter a question.")
        with s.lock:
            return jsonify(jsonsafe(_handle_question(s, question)))

    def _handle_question(s: Session, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        route: Dict[str, Any] = {"question": question, "at": time.strftime("%H:%M:%S")}
        plan = None
        try:
            plan = planner.plan(question, s)
            t_plan = time.perf_counter()
            route.update(operation=plan.operation, planned_target=plan.target_column)
            try:
                vp = validate_plan(plan, s.df, s.profile, s.roles)
            except ClarificationNeeded as c:
                s.pending_clarification = {"asked": c.question, "original_question": question, "options": c.options[:15]}
                s.turns.append({"question": question, "status": "clarification", "plan": plan.model_dump(exclude_none=True), "clarification": c.question})
                route.update(status="clarification")
                s.routing_log.append(route)
                log.info("ROUTE session=%s q=%r -> CLARIFY (%s)", s.id[:8], question, c.question)
                return {"type": "clarification", "answer": c.question, "options": c.options, "plan": plan.model_dump(exclude_none=True),
                        "result": None, "answer_source": "udi", "timing": {"plan_s": round(t_plan - t0, 2)}}
            s.pending_clarification = None
            executed = vp.plan.model_dump(exclude_none=True)
            result = execute(vp, s.ctx)
            t_exec = time.perf_counter()
            answer, source, note = explain_result(llm, question, vp.plan, result)
            t_end = time.perf_counter()
            model = (result.get("evaluation") or {}).get("selected_model")
            route.update(status="ok", operation=vp.plan.operation, target=vp.plan.target_column, group_by=vp.plan.group_by, time_column=vp.time_column if vp.plan.operation in ("get_current_state", "analyze_trend", "forecast") else None,
                         aggregation=vp.plan.aggregation, filters=vp.filters_applied, model=model, ml=result["provenance"]["uses_machine_learning"], seconds=round(t_end - t0, 2))
            s.routing_log.append(route)
            log.info("ROUTE session=%s q=%r -> op=%s target=%s group_by=%s model=%s ml=%s", s.id[:8], question, vp.plan.operation,
                     vp.plan.target_column, vp.plan.group_by, model, route["ml"])
            s.turns.append({"question": question, "status": "ok", "plan": executed, "digest": digest(result)})
            return {"type": "answer", "answer": answer, "answer_source": source, "explanation_note": note, "plan": executed, "result": result,
                    "timing": {"plan_s": round(t_plan - t0, 2), "execute_s": round(t_exec - t_plan, 2), "explain_s": round(t_end - t_exec, 2)}}
        except UDIError as e:
            route.update(status="error", error=e.code, seconds=round(time.perf_counter() - t0, 2))
            s.routing_log.append(route)
            s.turns.append({"question": question, "status": "error", "plan": plan.model_dump(exclude_none=True) if plan else {}, "error": f"{e.code}: {e.message[:200]}"})
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
