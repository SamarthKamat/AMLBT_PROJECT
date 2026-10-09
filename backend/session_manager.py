"""In-memory sessions: dataset, profile, conversation context and per-session model cache."""
from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from config import MAX_SESSIONS, SESSION_TTL_SECONDS
from context import ExecContext


@dataclass
class Session:
    id: str
    domain: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    profile: Optional[Dict[str, Any]] = None
    llm_profile: Optional[Dict[str, Any]] = None
    roles: Dict[str, Any] = field(default_factory=dict)
    ctx: Optional[ExecContext] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    routing_log: List[Dict[str, Any]] = field(default_factory=list)
    pending_clarification: Optional[Dict[str, Any]] = None
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def reset_conversation(self) -> None:
        self.turns.clear()
        self.routing_log.clear()
        self.pending_clarification = None
        if self.ctx:
            self.ctx.cache.clear()


class SessionStore:
    def __init__(self, ttl: int = SESSION_TTL_SECONDS, max_sessions: int = MAX_SESSIONS):
        self._s: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self.ttl, self.max_sessions = ttl, max_sessions

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, v in self._s.items() if now - v.touched > self.ttl]:
            del self._s[sid]
        while len(self._s) > self.max_sessions:
            oldest = min(self._s.values(), key=lambda v: v.touched)
            del self._s[oldest.id]

    def create(self, domain: Optional[str] = None) -> Session:
        with self._lock:
            self._evict()
            s = Session(id=uuid.uuid4().hex, domain=domain)
            self._s[s.id] = s
            return s

    def get(self, sid: Optional[str]) -> Optional[Session]:
        if not sid:
            return None
        with self._lock:
            s = self._s.get(sid)
            if s:
                s.touched = time.time()
            return s

    def delete(self, sid: Optional[str]) -> None:
        with self._lock:
            self._s.pop(sid or "", None)
