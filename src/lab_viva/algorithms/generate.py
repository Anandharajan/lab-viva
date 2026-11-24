"""Grounded question generation with a hallucination guard (A3).

Each question targets a (concept, Bloom level) pair with a source passage; the answer key is
taken from the manual. A question is accepted only if max_s P_entail(answer | s) >= 0.7 over
manual passages. The default entailment scorer is a lexical-support proxy (content-word recall
of the answer in the passage); plug in a local NLI cross-encoder for the paper's evaluation.
An optional local LLM (lab_viva.io.llm) may *propose* questions; grounding and faculty approval still apply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable

from .concepts import STOP, Concept

Entail = Callable[[str, str], float]

TEMPLATES = {
    "Remember": ["What is the {c} used for in this experiment?", "Name the role of the {c} in the setup."],
    "Understand": ["Explain the {c} as used in this experiment.", "Describe how the {c} affects the observation."],
    "Apply": ["How would you use the {c} to obtain the required reading?", "Using the {c}, how do you obtain the result?"],
    "Analyse": ["What happens to the result if the {c} is changed? Why?", "Compare the expected and observed behaviour of the {c}."],
}
FORMULA_TEMPLATES = {
    "Apply": ["How do you use the relation {c} to compute the result in this experiment?"] * 2,
    "Analyse": ["What does the relation {c} tell you, and what would change its value?"] * 2,
}
_HEADER = re.compile(r"^\s*(experiment\s*\d*|aim|apparatus( required)?|theory|procedure|result|conclusion)\s*[:.-]\s*", re.I)


def answer_sentence(c: Concept, sents: list[str], k: int) -> tuple[int, str] | None:
    """Pick an informative manual sentence for the answer key: skip titles and bare lists,
    strip leading 'Theory:' style labels, prefer definition-style sentences with more content.
    Returns None when no usable sentence is left for this slot (the question is then skipped)."""
    name = re.escape(c.name.lower())

    def info(i: int) -> int:
        s = sents[i]
        body = _HEADER.sub("", s)
        if re.match(r"^\s*experiment\b", s, re.I):
            return -100  # a title line is never an answer
        if body.count(",") >= 3:
            return -50  # an apparatus list names the item but does not explain it
        score = len([w for w in re.findall(r"[a-z0-9]+", body.lower()) if w not in STOP and len(w) > 2])
        if re.match(r"^\s*aim\b", s, re.I):
            score -= 5
        if re.search(rf"\b{name}\s+(is|are|means|refers|gives|measures|conducts)\b", body.lower()):
            score += 10  # definition-style sentence
        return score

    usable = sorted((i for i in c.sentences if info(i) > 0), key=lambda i: (-info(i), i))
    if k >= len(usable):
        return None
    i = usable[k]
    return i, _HEADER.sub("", sents[i]).strip()


KIND_LEVELS = {"equipment": ["Remember", "Apply"], "formula": ["Apply", "Analyse"], "term": ["Understand", "Analyse"]}


@dataclass
class Question:
    qid: str
    concept: str
    bloom: str
    text: str
    answer: str
    source: int  # sentence index in the manual
    support: float


def lexical_entail(answer: str, passage: str) -> float:
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if w not in STOP and len(w) > 2]
    if not words:
        return 0.0
    pw = set(re.findall(r"[a-z0-9]+", passage.lower()))
    return sum(w in pw for w in words) / len(words)


def grounded(answer: str, passages: list[str], entail: Entail = lexical_entail, thr: float = 0.7) -> float:
    best = max((entail(answer, p) for p in passages), default=0.0)
    return best if best >= thr else 0.0


def build_bank(concepts: list[Concept], sents: list[str], experiment: str = "E1",
               entail: Entail = lexical_entail, proposer: Callable[[str, str, str], tuple[str, str]] | None = None
               ) -> tuple[list[Question], list[dict[str, str]]]:
    bank, rejected = [], []
    for c in concepts:
        for k, lv in enumerate(KIND_LEVELS[c.kind]):
            picked = answer_sentence(c, sents, k)
            if picked is None:
                continue  # no informative manual sentence for this slot: skip rather than give a bad key
            src, answer = picked
            text = (FORMULA_TEMPLATES if c.kind == "formula" else TEMPLATES)[lv][k % 2].format(c=c.label)
            qid = f"{experiment}-{len(bank) + len(rejected) + 1:03d}"
            if proposer is not None:
                try:
                    text, answer = proposer(c.label, lv, sents[src])
                except Exception as e:  # noqa: BLE001 - one bad LLM reply must not stop the bank
                    rejected.append({"qid": qid, "text": "", "reason": f"LLM proposal failed: {e}"})
                    continue
            sup = grounded(answer, sents, entail)
            if sup > 0:
                bank.append(Question(qid, c.name, lv, text, answer, src, round(sup, 3)))
            else:
                rejected.append({"qid": qid, "text": text, "reason": "answer not entailed by the manual"})
    return bank, rejected
