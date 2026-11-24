"""Local-LLM path, tested against a stand-in OpenAI-compatible server on 127.0.0.1 (no network,
no model download). The stand-in returns one grounded answer, one hallucinated answer and one
malformed reply, so the grounding guard and error handling are exercised end to end."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lab_viva import api
from lab_viva.algorithms.concepts import extract
from lab_viva.algorithms.generate import build_bank, lexical_entail
from lab_viva.io.llm import ChatClient, LLMError, llm_proposer, parse_json_reply
from lab_viva.synth import EXAMPLE_MANUAL

REPLIES = [
    'Sure! ```json\n{"question": "What reduces the ripple factor?", '
    '"answer": "A filter capacitor across the load resistor reduces the ripple factor of the output."}\n```',
    '{"question": "Who invented the diode?", "answer": "Maxwell invented the diode in 1802 in Paris."}',
    "I cannot answer that.",
]


class Fake(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.calls.append(body)
        reply = REPLIES[(len(Fake.calls) - 1) % len(REPLIES)]
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Fake.calls = []
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_parse_json_reply():
    assert parse_json_reply('x {"question": "Q?", "answer": "A."} y') == {"question": "Q?", "answer": "A."}
    for bad in ("nothing", "{bad json}", '{"question": 1, "answer": "a"}'):
        with pytest.raises(LLMError):
            parse_json_reply(bad)


def test_remote_endpoint_needs_opt_in():
    with pytest.raises(ValueError, match="not a local endpoint"):
        ChatClient("https://api.example.com/v1")
    assert ChatClient("https://api.example.com/v1", allow_remote=True).url.endswith("/chat/completions")


def test_llm_bank_grounding_rejects_hallucination(server):
    concepts, _, sents = extract(EXAMPLE_MANUAL)
    client = ChatClient(server, model="test-model")
    bank, rejected = build_bank(concepts[:3], sents, entail=lexical_entail, proposer=llm_proposer(client, retries=0))
    reasons = [r["reason"] for r in rejected]
    assert any(q.text == "What reduces the ripple factor?" for q in bank)          # grounded -> kept
    assert "Who invented the diode?" in [r["text"] for r in rejected]            # hallucinated -> rejected
    assert any(r.startswith("LLM proposal failed") for r in reasons)             # malformed -> skipped, no crash
    assert Fake.calls[0]["model"] == "test-model" and Fake.calls[0]["messages"][0]["role"] == "user"


def test_unreachable_llm_gives_actionable_error():
    c = ChatClient("http://127.0.0.1:9/v1", timeout=2)
    with pytest.raises(LLMError, match="Is Ollama running"):
        c.complete("hi")


def test_api_build_with_llm(server, tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_VIVA_HOME", str(tmp_path))
    import lab_viva.io.llm as llm
    monkeypatch.setattr(llm, "DEFAULT_URL", server)
    monkeypatch.setattr(llm.ChatClient.__init__, "__defaults__", (server, "m", False, 120, 0.2))
    (tmp_path / "manual.txt").write_text(EXAMPLE_MANUAL)
    out = api.build_bank(str(tmp_path / "manual.txt"), llm=True, grounding="lexical")
    assert out["questions"] >= 1 and out["rejected_ungrounded"]
