"""Local LLM question proposer and NLI grounding scorer.

LLM: any OpenAI-compatible chat endpoint running on this machine -- Ollama
(http://127.0.0.1:11434/v1), llama.cpp `llama-server`, vLLM or LM Studio. Only loopback hosts
are allowed unless allow_remote=True, so lab manuals and answers stay on campus. The LLM only
*proposes* (question, answer); the answer must still be entailed by the manual (grounding
check) and approved by faculty before it is asked.

NLI: a local cross-encoder (cross-encoder/nli-deberta-v3-small) gives P(entailment) of the
answer given each manual sentence. Install with `pip install "lab-viva[nli]"`.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any
from collections.abc import Callable
from urllib.parse import urlparse

DEFAULT_URL = os.environ.get("LAB_VIVA_LLM_URL", "http://127.0.0.1:11434/v1")
DEFAULT_MODEL = os.environ.get("LAB_VIVA_LLM_MODEL", "llama3.2:3b")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

PROMPT = """You write oral-exam (viva) questions for an engineering lab course.
Write ONE {level}-level question about "{concept}" that can be answered ONLY from the passage,
and its short answer copied or closely paraphrased from the passage.
Reply with JSON only: {{"question": "...", "answer": "..."}}

Passage: {passage}"""


class LLMError(RuntimeError):
    pass


def parse_json_reply(text: str) -> dict[str, str]:
    """Extract the first JSON object from a model reply (models often wrap it in prose or ```)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON object in reply: {text[:120]!r}")
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON in reply: {e}") from e
    if not isinstance(d.get("question"), str) or not isinstance(d.get("answer"), str):
        raise LLMError("reply must contain string fields 'question' and 'answer'")
    return {"question": d["question"].strip(), "answer": d["answer"].strip()}


class ChatClient:
    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, allow_remote: bool = False,
                 timeout: float = 120, temperature: float = 0.2) -> None:
        host = urlparse(base_url).hostname or ""
        if host not in LOOPBACK and not allow_remote:
            raise ValueError(f"{base_url} is not a local endpoint; pass allow_remote=True to send manual text there")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.temperature = model, timeout, temperature
        self.key = os.environ.get("LAB_VIVA_LLM_KEY", "")

    def complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, body, headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data: dict[str, Any] = json.loads(r.read())
        except OSError as e:
            raise LLMError(f"cannot reach the local LLM at {self.url} ({e}). Is Ollama running? "
                           "Try: ollama serve  &&  ollama pull " + self.model) from e
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:120]}") from e


def llm_proposer(client: ChatClient, retries: int = 1) -> Callable[[str, str, str], tuple[str, str]]:
    def propose(concept: str, level: str, passage: str) -> tuple[str, str]:
        last: Exception | None = None
        for _ in range(retries + 1):
            try:
                d = parse_json_reply(client.complete(PROMPT.format(level=level, concept=concept, passage=passage)))
                return d["question"], d["answer"]
            except LLMError as e:
                last = e
        raise LLMError(f"{concept}/{level}: {last}")
    return propose


class NLIEntail:
    """P(entailment) of a hypothesis (the answer) given a premise (a manual sentence)."""

    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-small") -> None:
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model)
        labels = {str(v).lower(): int(k) for k, v in self.ce.model.config.id2label.items()}
        self.ie = labels["entailment"]
        self._cache: dict[tuple[str, str], float] = {}

    def __call__(self, hypothesis: str, premise: str) -> float:
        key = (hypothesis, premise)
        if key not in self._cache:
            import numpy as np
            z = np.asarray(self.ce.predict([(premise, hypothesis)], show_progress_bar=False), float)[0]
            p = np.exp(z - z.max())
            self._cache[key] = float(p[self.ie] / p.sum())
        return self._cache[key]


def get_entail(kind: str = "auto") -> Callable[[str, str], float]:
    from ..algorithms.generate import lexical_entail
    if kind == "lexical":
        return lexical_entail
    try:
        return NLIEntail()
    except ImportError:
        if kind == "nli":
            raise SystemExit('NLI grounding needs: pip install "lab-viva[nli]"') from None
        import sys
        print('[lab-viva] sentence-transformers not installed: lexical grounding proxy in use. '
              'For NLI: pip install "lab-viva[nli]"', file=sys.stderr)
        return lexical_entail
