# Contributing to auto-swe-agent

Thanks for your interest in contributing! 🚀

## Development Setup

```bash
git clone https://github.com/YashKasare21/auto-swe-agent.git
cd auto-swe-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_main.py -v --tb=short
```

## Code Style

This project uses:

- **[Black](https://github.com/psf/black)** for formatting
- **[isort](https://github.com/PyCQA/isort)** for import sorting
- **[mypy](https://mypy.readthedocs.io/)** for type checking

```bash
# Format code
make format

# Check formatting
make lint
```

## Adding a New Agent

1. Create `agents/your_agent.py` with a node function and system prompt.
2. The node function receives `state: GraphState` and returns a dict.
3. Use `from agents.base import invoke_agent` for LLM calls.
4. Register the node and routing in `agent.py`.

### Agent template

```python
# agents/my_agent.py
MY_SYSTEM = "You are a specialized agent that..."

def my_agent_node(state: dict) -> dict:
    result = invoke_agent(MY_SYSTEM, state, "my_agent")
    return result
```

## Adding a New Tool

1. Create a function in `tools/your_tool.py` decorated with `@tool`.
2. Add it to the `tools` list in `agent.py`.
3. If it needs tracing, use `@trace_tool("your_tool_name")`.

## Architecture Decisions

- **Multi-agent by default** — the `--single-agent` flag exists for A/B comparison only.
- **Each agent uses a different model** — configured via `FALLBACK_MODELS` per agent in `agents/base.py`.
- **Functional routing** — routing decisions are pure functions operating on `GraphState`.
- **Docker sandbox** — all bash commands run in containers, never on the host.

## Pull Request Process

1. Fork the repo and create a feature branch.
2. Run `make format` and `make lint` before committing.
3. Add tests for new functionality.
4. Update `README.md` if adding visible features.
5. Submit a PR with a clear description of the changes.

## Need Help?

Open an issue or start a discussion on GitHub.
