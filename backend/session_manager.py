"""Sessions: dataset, profile, semantics, conversation context and per-session model cache.

Phase 2: sessions have an owner and can be persisted through storage.Storage, so a restart of the server does not
lose uploaded datasets or the conversation; trained models are not persisted (they are retrained on demand).
The original upload is kept next to any cleaned version so data cleaning can be reverted.
"""
from __future__ import annotations
import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from config import MAX_SESSIONS, SESSION_TTL_SECONDS
from context import ExecContext

log = logging.getLogger("udi.sessions")


@dataclass
class Session:
    id: str
    domain: Optional[str] = None
    owner: str = "local"
    df: Optional[pd.DataFrame] = None
    original_df: Optional[pd.DataFrame] = None
    profile: Optional[Dict[str, Any]] = None
    llm_profile: Optional[Dict[str, Any]] = None
    roles: Dict[str, Any] = field(default_factory=dict)
    ctx: Optional[ExecContext] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    routing_log: List[Dict[str, Any]] = field(default_factory=list)
    pending_clarification: Optional[Dict[str, Any]] = None
    cleaning_log: List[Dict[str, Any]] = field(default_factory=list)
    dataset_name: Optional[str] = None
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def reset_conversation(self) -> None:
        self.turns.clear()
        self.routing_log.clear()
        self.pending_clarification = None
        if self.ctx:
            self.ctx.cache.clear()

    def set_dataset(self, df: pd.DataFrame, name: str, original: Optional[pd.DataFrame] = None, content_id: Optional[str] = None) -> None:
        """Profile the data and (re)build everything derived from it. Conversation and model cache are reset."""
        from data_profiler import profile_dataframe, profile_for_llm
        from relationships import analyse_relationships, relationships_digest
        from semantics import infer_semantics
        from validator import roles_from_profile
        profile = profile_dataframe(df, name)
        roles = roles_from_profile(profile)
        sem = infer_semantics(df, profile)
        did = content_id or hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]
        ctx = ExecContext(roles=roles, profile=profile, dataset_id=did, semantics=sem, full_data=df)
        rel = analyse_relationships(df, profile, sem)
        ctx.artifacts["relationships"] = rel
        self.df, self.profile, self.roles, self.ctx, self.dataset_name = df, profile, roles, ctx, name
        if original is not None:
            self.original_df = original
        self.llm_profile = profile_for_llm(profile, df, config.GEMINI_SAMPLE_ROWS, sem, relationships_digest(rel))
        self.reset_conversation()


class SessionStore:
    def __init__(self, ttl: int = SESSION_TTL_SECONDS, max_sessions: int = MAX_SESSIONS, storage=None):
        self._s: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self.ttl, self.max_sessions = ttl, max_sessions
        self.storage = storage

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, v in self._s.items() if now - v.touched > self.ttl]:
            del self._s[sid]
            if self.storage:
                self.storage.delete_session(sid)
        while len(self._s) > self.max_sessions:          # memory cap: persisted sessions are reloaded on demand
            oldest = min(self._s.values(), key=lambda v: v.touched)
            del self._s[oldest.id]

    def create(self, domain: Optional[str] = None, owner: str = "local") -> Session:
        with self._lock:
            self._evict()
            s = Session(id=uuid.uuid4().hex, domain=domain, owner=owner)
            self._s[s.id] = s
            if self.storage:
                self.storage.save_session(s.id, owner, domain, None)
            return s

    def get(self, sid: Optional[str]) -> Optional[Session]:
        if not sid:
            return None
        with self._lock:
            s = self._s.get(sid)
            if s is None and self.storage:
                s = self._restore(sid)
            if s:
                s.touched = time.time()
            return s

    def _restore(self, sid: str) -> Optional[Session]:
        row = self.storage.session_row(sid)
        if not row or time.time() - (row["touched"] or 0) > self.ttl:
            return None
        s = Session(id=sid, domain=row["domain"], owner=row["owner"])
        try:
            df, orig = self.storage.load_dataset(sid)
            if df is not None:
                s.set_dataset(df, row["dataset_name"] or "dataset", original=orig)
                s.turns = self.storage.turns(sid, 4)
        except Exception as exc:                          # a corrupt snapshot must not take the server down
            log.warning("Could not restore session %s: %s", sid, exc)
            return None
        self._s[sid] = s
        log.info("Restored session %s from storage", sid[:8])
        return s

    def persist_dataset(self, s: Session, cleaned: bool = False) -> None:
        if self.storage and s.df is not None:
            self.storage.save_dataset(s.id, s.df, s.original_df)
            self.storage.save_session(s.id, s.owner, s.domain, s.dataset_name, cleaned)
            self.storage.clear_conversation(s.id)

    def delete(self, sid: Optional[str]) -> None:
        with self._lock:
            self._s.pop(sid or "", None)
            if self.storage and sid:
                self.storage.delete_session(sid)

    def active(self) -> int:
        return len(self._s)
