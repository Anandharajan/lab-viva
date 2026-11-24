# Lab-Viva: results and honest status

All numbers come from the automated tests (`python -m pytest -q`) on simulated students.

| Check | Data | Result |
|---|---|---|
| Item difficulty recovery (2PL) | 400 simulated students × 20 items | corr(b̂, b) > 0.95; Spearman(â, a) > 0.4; corr(θ̂, θ) > 0.85 |
| Adaptive (CAT) vs random 8 items | 200 simulated students, 40 items | lower mean ability error |
| Grounding guard | LLM stand-in returning a grounded, a hallucinated and a malformed reply | grounded kept; hallucinated rejected; malformed skipped without stopping the bank |
| Approval gate | unapproved bank | nothing is asked until faculty approve |

## Gaps and caveats

- **The local-LLM path has not been run against a real model in this environment** (no Ollama installed here). It is
  tested end to end against a stand-in OpenAI-compatible server. Run `ollama pull llama3.2:3b` and
  `lab-viva build manual.txt --llm` to try it for real, and report question validity before claiming it.
- The template (no-LLM) questions are deliberately simple; question validity ("answerable from the manual", rated by
  2 faculty) and the 2-lab-cycle IRT check need the classroom pilot.
- NLI grounding uses `cross-encoder/nli-deberta-v3-small`; the lexical proxy is used only when that model is not installed.
