"""Public API (CLI, MCP). Questions must be approved by faculty before they are asked."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from . import roots
from .algorithms import cat
from .algorithms.concepts import extract
from .algorithms.generate import build_bank as _build
from .algorithms.irt import ItemParams, map_update
from .store import connect

_SESSIONS: dict[tuple[str, str], cat.Session] = {}
_PENDING: dict[tuple[str, str], str] = {}


def _manual_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":  # pragma: no cover - optional dependency
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if path.suffix.lower() == ".docx":  # pragma: no cover - optional dependency
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8")


def _items(experiment: str) -> list[cat.Item]:
    with connect() as con:
        rows = con.execute("SELECT qid, concept, a, b FROM items WHERE experiment=? AND approved=1", (experiment,)).fetchall()
    return [cat.Item(r["qid"], r["concept"], r["a"], r["b"]) for r in rows]


# ---------------- MCP tools ----------------

def build_bank(manual: str, experiment: str = "E1", llm: bool = False, llm_model: str | None = None,
               grounding: str = "auto") -> dict[str, Any]:
    """Extract a concept graph from a lab manual and generate grounded viva questions (pending approval)."""
    concepts, edges, sents = extract(_manual_text(roots.check(manual)))
    from .io.llm import DEFAULT_MODEL, ChatClient, get_entail, llm_proposer
    proposer = llm_proposer(ChatClient(model=llm_model or DEFAULT_MODEL)) if llm else None
    bank, rejected = _build(concepts, sents, experiment, entail=get_entail(grounding), proposer=proposer)
    with connect() as con:
        for q in bank:
            ip = ItemParams.cold(q.bloom)
            con.execute("INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?,?,0,?,?,?,?)",
                        (q.qid, experiment, q.concept, q.bloom, q.text, q.answer, q.source, q.support,
                         ip.a, ip.b, ip.b_prior, ip.se_b))
    return {"concepts": [c.name for c in concepts], "edges": len(edges), "questions": len(bank),
            "rejected_ungrounded": rejected}


def list_bank(experiment: str = "E1") -> list[dict[str, Any]]:
    """Questions with answer key, source sentence, Bloom level, approval and IRT parameters."""
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM items WHERE experiment=? ORDER BY qid", (experiment,))]


def approve(qids: list[str]) -> int:
    """Faculty approval of generated questions (required before they can be asked)."""
    with connect() as con:
        return con.executemany("UPDATE items SET approved=1 WHERE qid=?", [(q,) for q in qids]).rowcount


def start_viva(student: str, experiment: str = "E1", batch: str = "B1") -> dict[str, Any]:
    """Start an adaptive viva; returns the first question (max information, unseen concept, unused in batch)."""
    _SESSIONS[(batch, student)] = cat.Session(student)
    return _ask(student, experiment, batch)


def _ask(student: str, experiment: str, batch: str) -> dict[str, Any]:
    s = _SESSIONS[(batch, student)]
    with connect() as con:
        used = {r[0] for r in con.execute("SELECT qid FROM responses WHERE batch=? AND student<>?", (batch, student))}
    it = cat.next_item(_items(experiment), s, used)
    th, se = s.theta
    if it is None:
        return {"done": True, "theta": th, "se": se, "asked": s.asked}
    _PENDING[(batch, student)] = it.iid
    with connect() as con:
        q = con.execute("SELECT text, answer, concept, bloom FROM items WHERE qid=?", (it.iid,)).fetchone()
    return {"done": False, "qid": it.iid, "question": q["text"], "answer_key": q["answer"],
            "concept": q["concept"], "bloom": q["bloom"], "theta": th, "se": se}


def record(student: str, correct: bool, experiment: str = "E1", batch: str = "B1") -> dict[str, Any]:
    """Record the faculty's right/wrong mark for the pending question and return the next question."""
    s = _SESSIONS[(batch, student)]
    qid = _PENDING.pop((batch, student))
    it = next(i for i in _items(experiment) if i.iid == qid)
    cat.record(s, it, correct)
    with connect() as con:
        con.execute("INSERT INTO responses(batch, student, qid, correct, theta) VALUES (?,?,?,?,?)",
                    (batch, student, qid, int(correct), s.theta[0]))
    return _ask(student, experiment, batch)


def recalibrate(experiment: str = "E1") -> int:
    """Online MAP update of item difficulty/discrimination from all recorded responses."""
    with connect() as con:
        rows = con.execute("SELECT r.qid, r.correct, r.theta FROM responses r JOIN items i ON r.qid=i.qid "
                           "WHERE i.experiment=?", (experiment,)).fetchall()
        by: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for r in rows:
            by[r["qid"]].append((r["theta"], r["correct"]))
        for qid, data in by.items():
            it = con.execute("SELECT a, b, b_prior, se_b FROM items WHERE qid=?", (qid,)).fetchone()
            new = map_update(ItemParams(it["a"], it["b"], it["b_prior"], it["se_b"]), data)
            con.execute("UPDATE items SET a=?, b=?, se_b=? WHERE qid=?", (new.a, new.b, new.se_b, qid))
    return len(by)


def report(batch: str = "B1") -> dict[str, Any]:
    """Per-student ability, concept gaps and question exposure rates for a batch."""
    with connect() as con:
        rows = con.execute("SELECT r.student, r.qid, r.correct, r.theta, i.concept FROM responses r "
                           "JOIN items i ON r.qid=i.qid WHERE r.batch=? ORDER BY r.id", (batch,)).fetchall()
    students: dict[str, dict[str, Any]] = {}
    exposure: dict[str, int] = defaultdict(int)
    for r in rows:
        d = students.setdefault(r["student"], {"theta": 0.0, "wrong_concepts": []})
        d["theta"] = r["theta"]
        if not r["correct"]:
            d["wrong_concepts"].append(r["concept"])
        exposure[r["qid"]] += 1
    n = max(1, len(students))
    return {"students": students, "exposure_rate": {q: c / n for q, c in exposure.items()}}


TOOLS = [build_bank, list_bank, approve, start_viva, record, recalibrate, report]
