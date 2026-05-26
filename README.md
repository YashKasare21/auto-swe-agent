# auto-swe-agent

An autonomous software engineering agent that reads a natural language issue, explores a codebase, implements the fix, and verifies it by running tests — all without human intervention.

Built with LangGraph, LiteLLM, and Docker.

---

## How it works

```
Issue (natural language)
        │
        ▼
  ┌─────────────┐
  │   Planner   │  ← LLM with tool bindings (Gemini / Llama via LiteLLM)
  └──────┬──────┘
         │ tool calls
         ▼
  ┌─────────────┐
  │  Executor   │  ← runs tools: list_files, read_file, search_codebase,
  └──────┬──────┘               write_to_file, run_bash_command
         │
         └──── loops until tests pass or iteration limit reached
```

The agent runs all bash commands inside an isolated **Docker sandbox** (volume-mounted to the workspace), so it can install packages, run pytest, and verify its own changes safely.

---

## Features

- **Agentic loop** — LangGraph state machine with planner → executor → planner cycling
- **Docker sandbox** — isolated `python:3.11-slim` container; no commands run on your host
- **Multi-model fallback** — tries `gemini-2.0-flash → gemini-2.0-flash-lite → llama-3.3-70b → llama3-8b` in order; skips models with missing API keys or rate limit errors
- **Hallucination guard** — if the agent claims it's done without writing any files, it's forced back to the planner
- **Context management** — trims to last 10 messages; truncates tool outputs at 4000 chars to stay within token limits
- **Eval harness** — golden test cases with auto-reset, agent subprocess runner, and JSON result logging

---

## Project structure

```
auto-swe-agent/
├── agent.py          # Core agent: LangGraph graph, tools, Docker sandbox, fallback chain
├── main.py           # Sample target app (FastAPI /add endpoint) used in eval
├── test_main.py      # Tests for the sample target app
├── requirements.txt  # Python dependencies
├── eval/
│   └── run_eval.py   # Eval harness: defines cases, resets state, runs agent, reports results
├── docstream/        # Stub PDF library used as a target codebase in eval
│   ├── core/
│   │   ├── extractor.py      # PDFExtractor class
│   │   ├── extractor_v2.py   # extract_structured() with password support
│   │   └── generator.py      # Document continuation and merge logic
│   └── tests/
└── tests/            # Integration tests for eval cases
```

---

## Quickstart

**Prerequisites:** Python 3.11+, Docker running

```bash
git clone https://github.com/YashKasare21/auto-swe-agent.git
cd auto-swe-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Set at least one API key:

```bash
export GEMINI_API_KEY=your_key   # or GOOGLE_API_KEY
# or
export GROQ_API_KEY=your_key
```

Run the agent on any task:

```bash
python agent.py "The /add endpoint returns string concatenation instead of integer addition. Fix it so tests pass." --workspace ./
```

---

## Eval

The eval harness runs golden test cases end-to-end: resets the workspace to a buggy state, runs the agent, then validates with pytest.

```bash
python eval/run_eval.py
```

Current eval cases:

| Case | Description |
|------|-------------|
| `add-endpoint-bug` | Fix FastAPI `/add` endpoint (string concat → int addition) |
| `docstream-password-pdf` | Add password-protected PDF support + write a pytest test |

---

## Tech stack

| Component | Library |
|-----------|---------|
| Agent loop | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM routing | [LiteLLM](https://github.com/BerriAI/litellm) |
| LLM interface | [LangChain](https://github.com/langchain-ai/langchain) |
| Sandbox | [Docker SDK for Python](https://docker-py.readthedocs.io/) |
| Target app | [FastAPI](https://fastapi.tiangolo.com/) |
| Testing | [pytest](https://pytest.org/) |

---

## License

MIT
