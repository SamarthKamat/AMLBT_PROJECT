"""Operational controls (Phase 2): authentication, authorisation, rate limiting, background jobs, metrics.

Authentication (UDI_AUTH=on): users register/log in with a password (PBKDF2 hash via werkzeug); the server issues a
random bearer token and stores only its SHA-256 hash, with an expiry. The first registered user is an admin.
Authorisation: a session belongs to the user who created it; analysts can only reach their own sessions; admins can
also read /api/metrics and list all sessions. With UDI_AUTH=off (local single-user default) every request acts as the
local user "local" (admin), which is appropriate only on a trusted machine.
"""
from __future__ import annotations
import hashlib
import secrets
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash

from errors import UDIError

TOKEN_TTL = 7 * 24 * 3600


# ----------------------------------------------------------------------------- auth
class Auth:
    def __init__(self, storage, enabled: bool):
        self.storage, self.enabled = storage, enabled

    @staticmethod
    def _h(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def register(self, username: str, password: str) -> Dict[str, Any]:
        username = (username or "").strip().lower()
        if not (3 <= len(username) <= 40) or not username.replace("_", "").replace(".", "").isalnum():
            raise UDIError("BAD_REQUEST", "Username must be 3-40 letters, digits, '.' or '_'.")
        if len(password or "") < 8:
            raise UDIError("BAD_REQUEST", "Password must be at least 8 characters.")
        if self.storage.execute("SELECT id FROM users WHERE username=?", (username,)):
            raise UDIError("BAD_REQUEST", "That username is taken.")
        role = "admin" if not self.storage.execute("SELECT id FROM users LIMIT 1") else "analyst"
        self.storage.execute("INSERT INTO users(username, password_hash, role, created) VALUES(?,?,?,?)",
                             (username, generate_password_hash(password), role, time.time()))
        return self.login(username, password)

    def login(self, username: str, password: str) -> Dict[str, Any]:
        rows = self.storage.execute("SELECT * FROM users WHERE username=?", ((username or "").strip().lower(),))
        if not rows or not check_password_hash(rows[0]["password_hash"], password or ""):
            raise UDIError("UNAUTHORIZED", "Wrong username or password.")
        token = secrets.token_urlsafe(32)
        now = time.time()
        self.storage.execute("INSERT INTO tokens(token_hash, user_id, created, expires) VALUES(?,?,?,?)", (self._h(token), rows[0]["id"], now, now + TOKEN_TTL))
        return {"token": token, "user": {"username": rows[0]["username"], "role": rows[0]["role"]}}

    def logout(self, token: Optional[str]) -> None:
        if token:
            self.storage.execute("DELETE FROM tokens WHERE token_hash=?", (self._h(token),))

    def user_for(self, header: Optional[str]) -> Dict[str, Any]:
        if not self.enabled:
            return {"username": "local", "role": "admin"}
        token = (header or "").removeprefix("Bearer ").strip()
        if not token:
            raise UDIError("UNAUTHORIZED", "Sign in to use UDI.")
        rows = self.storage.execute("SELECT u.username, u.role, t.expires FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token_hash=?", (self._h(token),))
        if not rows or rows[0]["expires"] < time.time():
            raise UDIError("UNAUTHORIZED", "Your sign-in has expired. Sign in again.")
        return {"username": rows[0]["username"], "role": rows[0]["role"]}


# ----------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding-window limit per (client, bucket): at most `limit` requests per `window` seconds."""

    def __init__(self, limits: Dict[str, int], window: float = 60.0):
        self.limits, self.window = limits, window
        self._hits: Dict[tuple, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, bucket: str) -> None:
        limit = self.limits.get(bucket)
        if not limit:
            return
        now = time.time()
        with self._lock:
            q = self._hits[(client, bucket)]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                raise UDIError("RATE_LIMITED", f"Too many {bucket} requests: at most {limit} per minute. Wait a moment and retry.", retryable=True)
            q.append(now)


# ----------------------------------------------------------------------------- background jobs
class JobQueue:
    """Small thread-pool job runner for long tasks (reports on large datasets). Results stay in memory for 1 hour."""

    def __init__(self, workers: int = 2, keep_seconds: int = 3600):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="udi-job")
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.keep = keep_seconds
        self._lock = threading.Lock()

    def submit(self, owner: str, kind: str, fn: Callable[[Callable[[float, str], None]], Any]) -> str:
        jid = uuid.uuid4().hex
        job = {"id": jid, "owner": owner, "kind": kind, "status": "queued", "progress": 0.0, "message": "queued", "created": time.time(), "result": None, "error": None}
        with self._lock:
            self._gc()
            self.jobs[jid] = job

        def progress(frac: float, msg: str):
            job.update(progress=round(float(frac), 3), message=msg)

        def run():
            job["status"] = "running"
            try:
                job["result"] = fn(progress)
                job.update(status="done", progress=1.0, message="done")
            except UDIError as e:
                job.update(status="error", error=e.to_dict())
            except Exception as e:                       # pragma: no cover
                job.update(status="error", error={"code": "BACKEND_ERROR", "error": f"{type(e).__name__}: {e}", "retryable": True, "trace": traceback.format_exc()[-800:]})
            job["finished"] = time.time()
        self.pool.submit(run)
        return jid

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(jid)

    def _gc(self):
        now = time.time()
        for k in [k for k, v in self.jobs.items() if v.get("finished") and now - v["finished"] > self.keep]:
            del self.jobs[k]


# ----------------------------------------------------------------------------- metrics
class Metrics:
    def __init__(self):
        self.started = time.time()
        self._lock = threading.Lock()
        self.requests: Dict[str, int] = defaultdict(int)
        self.errors: Dict[str, int] = defaultdict(int)
        self.operations: Dict[str, int] = defaultdict(int)
        self.planners: Dict[str, int] = defaultdict(int)
        self.latency: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    def request(self, endpoint: str, seconds: float, status: int):
        with self._lock:
            self.requests[endpoint] += 1
            self.latency[endpoint].append(seconds)
            if status >= 400:
                self.errors[str(status)] += 1

    def error_code(self, code: str):
        with self._lock:
            self.errors[code] += 1

    def answered(self, operation: str, planner: str):
        with self._lock:
            self.operations[operation] += 1
            self.planners[planner] += 1

    def snapshot(self, active_sessions: int) -> Dict[str, Any]:
        def pct(vals: List[float], q: float):
            if not vals:
                return None
            s = sorted(vals)
            return round(s[min(len(s) - 1, int(q * len(s)))], 3)
        with self._lock:
            lat = {k: {"p50_s": pct(list(v), .5), "p95_s": pct(list(v), .95), "n": len(v)} for k, v in self.latency.items()}
            return {"uptime_s": round(time.time() - self.started, 1), "active_sessions": active_sessions, "requests": dict(self.requests),
                    "errors": dict(self.errors), "operations": dict(self.operations), "planners": dict(self.planners), "latency": lat}
