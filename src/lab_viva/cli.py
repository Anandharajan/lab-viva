"""lab-viva command line (including an interactive faculty viva screen in the terminal)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import api
from .synth import EXAMPLE_MANUAL


def main(argv: list[str] | None = None, inp=input) -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(prog="lab-viva")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("manual"); b.add_argument("--experiment", default="E1")
    b.add_argument("--llm", action="store_true", help="let a local LLM (Ollama etc.) propose the questions")
    b.add_argument("--llm-model", help="model name on the local endpoint (env LAB_VIVA_LLM_MODEL)")
    b.add_argument("--grounding", choices=["auto", "nli", "lexical"], default="auto")
    ls = sub.add_parser("list"); ls.add_argument("--experiment", default="E1")
    ap_ = sub.add_parser("approve"); ap_.add_argument("qids", nargs="*"); ap_.add_argument("--all", action="store_true")
    ap_.add_argument("--experiment", default="E1")
    v = sub.add_parser("viva"); v.add_argument("student"); v.add_argument("--experiment", default="E1")
    v.add_argument("--batch", default="B1")
    r = sub.add_parser("report"); r.add_argument("--batch", default="B1")
    sub.add_parser("recalibrate")
    d = sub.add_parser("demo"); d.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "manual.txt").write_text(EXAMPLE_MANUAL, encoding="utf-8")
        print(a.out / "manual.txt")
    elif a.cmd == "build":
        print(json.dumps(api.build_bank(a.manual, a.experiment, a.llm, a.llm_model, a.grounding), indent=2))
    elif a.cmd == "list":
        for q in api.list_bank(a.experiment):
            print(f"{q['qid']} [{'x' if q['approved'] else ' '}] {q['bloom']:10} {q['text']}\n    key: {q['answer']}")
    elif a.cmd == "approve":
        ids = [q["qid"] for q in api.list_bank(a.experiment)] if a.all else a.qids
        print(f"approved {api.approve(ids)} question(s)")
    elif a.cmd == "viva":
        q = api.start_viva(a.student, a.experiment, a.batch)
        while not q["done"]:
            print(f"\n[{q['bloom']}] {q['question']}\n  key: {q['answer_key']}")
            ans = inp("correct? [y/n] ").strip().lower().startswith("y")
            q = api.record(a.student, ans, a.experiment, a.batch)
        print(f"ability {q['theta']:+.2f} (SE {q['se']:.2f}) after {len(q['asked'])} questions")
    elif a.cmd == "report":
        print(json.dumps(api.report(a.batch), indent=2))
    else:
        print(f"recalibrated {api.recalibrate()} item(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
