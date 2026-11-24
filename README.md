# Lab-Viva

**Fair, varied, outcome-mapped lab vivas — generated from your own lab manual.** Lab-Viva builds a grounded
question bank from the manual (optionally with a **local LLM**), rejects any answer the manual does not support,
and runs an **adaptive viva** that adjusts difficulty to each student with item response theory.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24502513.svg)](https://doi.org/10.5281/zenodo.24502513) [![CI](https://github.com/Anandharajan/lab-viva/actions/workflows/ci.yml/badge.svg)](https://github.com/Anandharajan/lab-viva/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anandharajan/lab-viva/blob/main/notebooks/quickstart.ipynb) [![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-black?logo=github)](https://codespaces.new/Anandharajan/lab-viva) [![GitHub stars](https://img.shields.io/github/stars/Anandharajan/lab-viva?style=social)](https://github.com/Anandharajan/lab-viva/stargazers)

> ⭐ **If Lab-Viva saves you viva-preparation time, please [star the repo](https://github.com/Anandharajan/lab-viva)** — it helps other faculty find it.

## Try it in 60 seconds

- **No install:** [open the Colab notebook](https://colab.research.google.com/github/Anandharajan/lab-viva/blob/main/notebooks/quickstart.ipynb) and press *Run all*.
- **In the browser with VS Code:** [open in GitHub Codespaces](https://codespaces.new/Anandharajan/lab-viva).
- **On your machine:**

```bash
pip install "lab-viva[nli] @ git+https://github.com/Anandharajan/lab-viva"
lab-viva demo example                 # writes an example rectifier lab manual
lab-viva build example/manual.txt     # concept graph + grounded questions
lab-viva approve --all                # faculty approval is required before anything is asked
lab-viva viva S001                    # adaptive viva in the terminal: mark each answer y/n
lab-viva report                       # ability per student, weak concepts, question exposure
```

Questions generated from the example manual (no LLM):

```
[Understand] Explain the ripple factor as used in this experiment.
     key: The ripple factor is the ratio of the RMS value of the AC component to the DC value of the output.
[Remember]   What is the multimeter used for in this experiment?
     key: Measure the DC output voltage using the multimeter.
[Analyse]    What does the relation r = 1.21 tell you, and what would change its value?
     key: For a half-wave rectifier the ripple factor r = 1.21 without a filter capacitor.
```

## Local LLM (on-premise)

Let a model on your own machine write the questions — manuals and answers never leave campus:

```bash
ollama pull llama3.2:3b                # or any model served by llama.cpp, vLLM or LM Studio
lab-viva build manual.txt --llm --grounding nli
```

Any OpenAI-compatible endpoint works (`LAB_VIVA_LLM_URL`, default `http://127.0.0.1:11434/v1`). Only loopback addresses
are allowed unless you opt in. The LLM only *proposes*: every answer must be entailed by the manual (local NLI
check), and every question needs faculty approval.

## How it works

- **Concept graph** from the manual: equipment, formulas, key terms; prerequisite and co-occurrence edges.
- **Grounding guard:** an answer is kept only if a local NLI model finds it entailed by a manual sentence.
- **2PL IRT:** difficulty starts from a Bloom-level prior and is re-estimated from real right/wrong marks.
- **Adaptive viva:** each next question maximises information at the student's current ability, covers new concepts
  first, and avoids questions already used in the same batch (so students cannot pass answers along).

## Use it from your AI assistant (MCP)

lab-viva ships a local [Model Context Protocol](https://modelcontextprotocol.io) server (stdio only, nothing leaves
your machine). Install with `pip install "lab-viva[mcp,nli] @ git+https://github.com/Anandharajan/lab-viva"`, then add this to
your client's MCP config (Claude Desktop, Cursor, VS Code and others use the same shape):

```json
{
  "mcpServers": {
    "lab-viva": { "command": "lab-viva-mcp" }
  }
}
```

Every tool call is written to a local audit log, and you can limit file access to chosen folders with the
`LAB_VIVA_ROOTS` environment variable.

## Status

Item-parameter recovery on simulated students (corr > 0.95); adaptive selection beats random 8-question vivas on
simulated data. A classroom pilot is planned — see [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Contributing

Bug reports, ideas and pull requests are welcome: [open an issue](https://github.com/Anandharajan/lab-viva/issues).
Run `python -m pytest -q` before sending a pull request.

## Citation

If you use lab-viva in research, please cite it (see [CITATION.cff](CITATION.cff), or GitHub's *Cite this repository* button).

## Licence

MIT © 2026 Anandharajan TRV, Department of ECE, St. Joseph College of Engineering, Chennai.

---

**If lab-viva saved you time, please ⭐ [star the repository](https://github.com/Anandharajan/lab-viva/stargazers)** — it is
the simplest way to help other researchers find it.
