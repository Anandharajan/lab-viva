"""Concept-graph extraction from a lab manual (rule-based; an optional local LLM may refine it,
and the faculty member always reviews it).

Concepts: equipment terms, formula left-hand sides, and repeated multi-word noun phrases.
Edges: co-occurrence in a sentence; prerequisite A->B when A is introduced before B and they co-occur.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

EQUIPMENT = ["oscilloscope", "multimeter", "function generator", "signal generator", "power supply",
             "breadboard", "resistor", "capacitor", "inductor", "diode", "transistor", "op-amp", "probe",
             "spectrum analyzer", "rheostat", "ammeter", "voltmeter", "crt", "dso", "ic 741", "logic analyzer"]
STOP = set("""a an the of and or to in on for with by is are be as at from this that these those it its into
using use used then than when where which while each every all any some such can will should must may your
note set connect observe record repeat step procedure aim apparatus required result theory""".split())


@dataclass
class Concept:
    name: str
    kind: str  # equipment | formula | term
    first: int  # first sentence index
    sentences: list[int] = field(default_factory=list)
    display: str = ""  # how the concept is shown in a question (e.g. "r = 1.21" for a formula)

    @property
    def label(self) -> str:
        return self.display or self.name


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if len(s.strip()) > 3]


def _phrases(s: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", s.lower())
    out, cur = [], []
    for w in words + ["."]:
        if w in STOP or w == ".":
            if 2 <= len(cur) <= 4:
                out.append(" ".join(cur))
            cur = []
        else:
            cur.append(w)
    return out


def extract(text: str, min_count: int = 2) -> tuple[list[Concept], list[tuple[str, str, str]], list[str]]:
    sents = sentences(text)
    found: dict[str, Concept] = {}

    def add(name: str, kind: str, i: int, display: str = "") -> None:
        c = found.setdefault(name, Concept(name, kind, i, display=display))
        if i not in c.sentences:
            c.sentences.append(i)

    counts = Counter(p for s in sents for p in set(_phrases(s)))
    for i, s in enumerate(sents):
        low = s.lower()
        for eq in EQUIPMENT:
            if re.search(rf"\b{re.escape(eq)}\b", low):
                add(eq, "equipment", i)
        for m in re.finditer(r"\b([A-Za-z][A-Za-z_]{0,12})\s*=\s*(?:-?\d+(?:\.\d+)?(?:\s*%)?|[^=,.;\d][^=,.;]{1,})", s):
            add(m.group(1), "formula", i, display=" ".join(m.group(0).split()))
        for p in _phrases(s):
            if counts[p] >= min_count:
                add(p, "term", i)
    concepts = sorted(found.values(), key=lambda c: c.first)
    edges = []
    for x in concepts:
        for y in concepts:
            if x.name < y.name and set(x.sentences) & set(y.sentences):
                edges.append((x.name, y.name, "co-occurs"))
            if x.first < y.first and set(x.sentences) & set(y.sentences):
                edges.append((x.name, y.name, "prerequisite"))
    return concepts, edges, sents
