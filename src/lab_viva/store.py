"""SQLite store (WAL) for question banks, approvals, responses and item parameters."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS items(qid TEXT PRIMARY KEY, experiment TEXT, concept TEXT, bloom TEXT, text TEXT,
  answer TEXT, source INTEGER, support REAL, approved INTEGER DEFAULT 0, a REAL, b REAL, b_prior REAL, se_b REAL);
CREATE TABLE IF NOT EXISTS responses(id INTEGER PRIMARY KEY, batch TEXT, student TEXT, qid TEXT, correct INTEGER,
  theta REAL, ts REAL DEFAULT (julianday('now')));
"""


def db_path() -> Path:
    return Path(os.environ.get("LAB_VIVA_HOME", Path.home() / ".lab-viva")) / "viva.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con
