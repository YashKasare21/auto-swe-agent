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
├── agent.py           # Core agent: LangGraph graph, tools, Docker sandbox, fallback chain
├── main.py            # Sample target app (FastAPI /add endpoint) used in eval
├── test_main.py       # Tests for the sample target app
├── requirements.txt   # Python dependencies
├── agents/            # Multi-agent architecture
│   ├── base.py        #   Agent runtime: LLM invocation, fallback, cost tracking
│   ├── manager.py     #   Manager: task complexity analysis
│   ├── planner.py     #   Planner: implementation steps
│   ├── coder.py       #   Coder: code implementation
│   └── reviewer.py    #   Reviewer: LGTM / NEEDS_FIX
├── indexing/          # RAG semantic code search
│   ├── parser.py      #   AST-based code chunking
│   ├── embedder.py    #   sentence-transformers embeddings
│   ├── vector_store.py #   FAISS vector store
│   └── build_index.py #   Index builder (CLI + auto-build)
├── tools/             # Agent tools
│   ├── git_tools.py   #   Git operations (branch, commit, PR description)
│   └── semantic_search.py  # Semantic code search tool
├── swe_bench/         # SWE-bench Lite evaluation
│   ├── harness.py     #   Harness: dataset loading, workspace setup, agent runner, evaluation
│   └── run_swe_bench.py #   CLI entry point for SWE-bench evaluation
├── observability/     # Langfuse tracing
│   ├── langfuse_client.py  #   Langfuse client with graceful degradation
│   └── tool_tracing.py     #   Tool execution tracing
├── eval/
│   └── run_eval.py    # Golden eval harness
├── docstream/         # Stub PDF library used in eval
├── scripts/           # Utility scripts (cost reports, SWE-bench report, etc.)
└── tests/             # Integration tests
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

---

## SWE-bench Lite Evaluation

Evaluate auto-swe-agent against the [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) benchmark — 300 real GitHub issues from 12 popular Python repos.

### Run

```bash
# Evaluate on the first 10 tasks
python -m swe_bench.run_swe_bench --num-tasks 10

# Run specific tasks
python -m swe_bench.run_swe_bench --instance-ids django__django-11011 django__django-11039

# Full eval with budget and single-agent comparison
python -m swe_bench.run_swe_bench --num-tasks 50 --budget 3.0 --single-agent
```

### View results

```bash
# Generate markdown report
python scripts/swe_bench_report.py

# Save to file
python scripts/swe_bench_report.py -o SWE_BENCH_RESULTS.md
```

### How it works

1. **Dataset loading** — Loads `princeton-nlp/SWE-bench_Lite` test split via HuggingFace `datasets`.
2. **Workspace setup** — For each task, clones the repo at the pre-bug `base_commit` into a temp directory.
3. **Agent execution** — Runs `agent.py` on the issue description with the cloned repo as workspace. The agent uses its normal multi-agent pipeline (Manager → Planner → Coder → Reviewer) with Docker sandbox, self-verification, and Git workflow.
4. **Patch extraction** — After the agent finishes, `git diff HEAD` is extracted from the workspace.
5. **Evaluation** — The patch is applied to a fresh clone at the same commit, then the benchmark's `fail_to_pass` tests are run. If they pass, the patch is correct.

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--num-tasks` | `10` | Number of tasks (first N). Use `-1` for all 300. |
| `--instance-ids` | — | Specific tasks by ID (e.g. `django__django-11011`) |
| `--budget` | `5.0` | Dollar budget per agent run |
| `--single-agent` | `false` | Use single-agent mode instead of multi-agent |
| `--agent-timeout` | `1800` | Seconds per task (30 min default) |
| `--no-cleanup` | `false` | Don't delete temp workspaces after eval |

### Comparison

| Baseline | Score | Notes |
|----------|-------|-------|
| **auto-swe-agent** | *coming soon* | Run `python -m swe_bench.run_swe_bench --num-tasks 300` |
| Devin (early) | 13.86% | [Cognition blog](https://www.cognition.ai/blog/introducing-devin) |
| SWE-agent (default) | 12.47% | [arXiv:2405.15793](https://arxiv.org/abs/2405.15793) |
| SWE-agent (with RAG) | 18.20% | Same paper, with retrieval augmentation |
| OpenCode Interpreter | 23.17% | [OSLAB/OpenCodeInterpreter](https://github.com/OSLAB/OpenCodeInterpreter) |

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
