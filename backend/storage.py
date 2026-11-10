"""Persistence (Phase 2): SQLite metadata + on-disk dataset snapshots.

Tables
  users(id, username, password_hash, role, created)
  tokens(token_hash, user_id, created, expires)
  sessions(id, owner, domain, dataset_name, created, touched, cleaned)
  turns(id, session_id, question, status, plan, digest, planner, created)
  history(id, session_id, operation, title, summary, seconds, planner, created)

Datasets are stored as pickled DataFrames (pandas' own format, written and read only by this server) in
UDI_DATA_DIR/datasets/<session>.pkl, with the original upload kept as <session>.orig.pkl so cleaning can be reverted.
Deleting a session removes its rows and files (dataset deletion control). Only hashes of auth tokens are stored.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
                                 role TEXT NOT NULL DEFAULT 'analyst', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tokens(token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created REAL NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, owner TEXT NOT NULL, domain TEXT, dataset_name TEXT, created REAL, touched REAL,
                                    cleaned INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, question TEXT, status TEXT, plan TEXT,
                                 digest TEXT, planner TEXT, created REAL);
CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, operation TEXT, title TEXT, summary TEXT,
                                   seconds REAL, planner TEXT, created REAL);
CREATE INDEX IF NOT EXISTS ix_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS ix_history_session ON history(session_id);
"""


class Storage:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "udi.sqlite3"
        self._lock = threading.RLock()
        c = self._conn()
        try:
            c.executescript(SCHEMA)
        finally:
            c.close()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def execute(self, sql: str, args: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            c = self._conn()
            try:
                return list(c.execute(sql, args).fetchall())
            finally:
                c.close()

    # ---------------------------------------------------------------- sessions
    def save_session(self, sid: str, owner: str, domain: Optional[str], dataset_name: Optional[str], cleaned: bool = False) -> None:
        now = time.time()
        self.execute("INSERT INTO sessions(id, owner, domain, dataset_name, created, touched, cleaned) VALUES(?,?,?,?,?,?,?) "
                     "ON CONFLICT(id) DO UPDATE SET domain=excluded.domain, dataset_name=excluded.dataset_name, touched=excluded.touched, cleaned=excluded.cleaned",
                     (sid, owner, domain, dataset_name, now, now, int(cleaned)))

    def touch(self, sid: str) -> None:
        self.execute("UPDATE sessions SET touched=? WHERE id=?", (time.time(), sid))

    def session_row(self, sid: str) -> Optional[Dict[str, Any]]:
        rows = self.execute("SELECT * FROM sessions WHERE id=?", (sid,))
        return dict(rows[0]) if rows else None

    def sessions_for(self, owner: Optional[str]) -> List[Dict[str, Any]]:
        if owner is None:
            return [dict(r) for r in self.execute("SELECT * FROM sessions ORDER BY touched DESC LIMIT 200")]
        return [dict(r) for r in self.execute("SELECT * FROM sessions WHERE owner=? ORDER BY touched DESC LIMIT 200", (owner,))]

    def _paths(self, sid: str):
        safe = "".join(ch for ch in sid if ch.isalnum())
        return self.root / "datasets" / f"{safe}.pkl", self.root / "datasets" / f"{safe}.orig.pkl"

    def save_dataset(self, sid: str, df: pd.DataFrame, original: Optional[pd.DataFrame] = None) -> None:
        cur, orig = self._paths(sid)
        df.to_pickle(cur)
        if original is not None:
            original.to_pickle(orig)

    def load_dataset(self, sid: str):
        cur, orig = self._paths(sid)
        if not cur.exists():
            return None, None
        df = pd.read_pickle(cur)
        return df, (pd.read_pickle(orig) if orig.exists() else None)

    def delete_session(self, sid: str) -> None:
        for p in self._paths(sid):
            p.unlink(missing_ok=True)
        self.execute("DELETE FROM turns WHERE session_id=?", (sid,))
        self.execute("DELETE FROM history WHERE session_id=?", (sid,))
        self.execute("DELETE FROM sessions WHERE id=?", (sid,))

    def clear_conversation(self, sid: str) -> None:
        self.execute("DELETE FROM turns WHERE session_id=?", (sid,))

    def purge_older_than(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        old = [r["id"] for r in self.execute("SELECT id FROM sessions WHERE touched < ?", (cutoff,))]
        for sid in old:
            self.delete_session(sid)
        return len(old)

    # ---------------------------------------------------------------- turns / history
    def add_turn(self, sid: str, turn: Dict[str, Any], planner: Optional[str]) -> None:
        self.execute("INSERT INTO turns(session_id, question, status, plan, digest, planner, created) VALUES(?,?,?,?,?,?,?)",
                     (sid, turn.get("question"), turn.get("status"), json.dumps(turn.get("plan") or {}, default=str),
                      json.dumps({k: turn.get(k) for k in ("digest", "clarification", "error") if turn.get(k) is not None}, default=str), planner, time.time()))

    def turns(self, sid: str, limit: int = 4) -> List[Dict[str, Any]]:
        rows = self.execute("SELECT * FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?", (sid, limit))
        out = []
        for r in reversed(rows):
            extra = json.loads(r["digest"] or "{}")
            out.append({"question": r["question"], "status": r["status"], "plan": json.loads(r["plan"] or "{}"), **extra})
        return out

    def add_history(self, sid: str, operation: str, title: str, summary: str, seconds: float, planner: Optional[str]) -> None:
        self.execute("INSERT INTO history(session_id, operation, title, summary, seconds, planner, created) VALUES(?,?,?,?,?,?,?)",
                     (sid, operation, title, summary[:2000], seconds, planner, time.time()))

    def history(self, sid: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.execute("SELECT operation, title, summary, seconds, planner, created FROM history WHERE session_id=? ORDER BY id", (sid,))]
