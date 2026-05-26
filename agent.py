import os
import time
from typing import Annotated, Optional, TypedDict

import docker
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tracking.cost_tracker import CostTracker
from resilience.circuit_breaker import CircuitBreaker
from resilience.retry import with_retry
from ui.state_manager import AgentStateManager

docker_client = docker.from_env()
_sandbox: docker.models.containers.Container = None
_SANDBOX_LABEL = "auto-swe-agent-sandbox"

# Module-level cost tracker (singleton, like _sandbox).
# Kept outside GraphState because TypedDict can't hold arbitrary class instances.
_cost_tracker: CostTracker = CostTracker(budget_usd=5.0)

# Module-level circuit breaker and event log (reset each run in main()).
_circuit_breaker: CircuitBreaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300)
_circuit_events: list[str] = []  # human-readable log of open/close events per run

# Module-level state manager for UI live monitoring (reset at each main()).
_state_manager: AgentStateManager = AgentStateManager()


def get_sandbox(workspace_dir: str) -> docker.models.containers.Container:
    global _sandbox
    if _sandbox is not None:
        return _sandbox
    existing = docker_client.containers.list(filters={"label": f"role={_SANDBOX_LABEL}"})
    if existing:
        _sandbox = existing[0]
        print(f"[Docker] Reusing sandbox: {_sandbox.short_id}")
        return _sandbox
    abs_workspace = os.path.abspath(workspace_dir)
    _sandbox = docker_client.containers.run(
        "python:3.11-slim",
        command="sleep infinity",
        detach=True,
        remove=True,
        labels={"role": _SANDBOX_LABEL},
        volumes={abs_workspace: {"bind": "/workspace", "mode": "rw"}},
        working_dir="/workspace",
    )
    print(f"[Docker] Sandbox started: {_sandbox.short_id}")
    print("[Docker] Installing packages...")
    exit_code, output = _sandbox.exec_run(
        ["pip", "install", "fastapi", "httpx", "pytest", "uvicorn"],
        demux=False,
    )
    if exit_code != 0:
        raise RuntimeError(f"[Docker] pip install failed (exit {exit_code}):\n{output.decode()}")
    exit_code, _ = _sandbox.exec_run(["python", "-c", "import fastapi, pytest, httpx, uvicorn"])
    if exit_code != 0:
        raise RuntimeError("[Docker] Health check failed — packages not importable after install.")
    print("[Docker] Sandbox ready.")
    return _sandbox


IGNORE_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules", ".next"}


@tool
def list_files(directory: str) -> str:
    """Return a directory tree string, ignoring common non-essential directories."""
    lines = []
    abs_dir = os.path.abspath(directory)
    for root, dirs, files in os.walk(abs_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        level = len(os.path.relpath(root, abs_dir).split(os.sep)) - 1 if root != abs_dir else 0
        lines.append(f"{'  ' * level}{os.path.basename(root) or root}/")
        for f in files:
            lines.append(f"{'  ' * (level + 1)}{f}")
    return "\n".join(lines) if lines else "Directory is empty or does not exist."


@tool
def read_file(filepath: str) -> str:
    """Return file contents, truncated at 2000 lines with a warning if exceeded."""
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()
    if len(lines) > 2000:
        return "".join(lines[:2000]) + "\n\n[WARNING: File truncated at 2000 lines to save context window.]"
    return "".join(lines)


@tool
def search_codebase(keyword: str, directory: str) -> str:
    """Search for a keyword in all files under directory, returning file:line matches."""
    matches = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if keyword in line:
                            matches.append(f"{fpath}:{i}: {line.rstrip()}")
            except OSError:
                pass
    return "\n".join(matches) if matches else "No matches found."


@tool
def write_to_file(filepath: str, content: str) -> str:
    """Write content to a file on the local filesystem (synced to the Docker sandbox via volume mount)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, "w") as f:
        f.write(content)
    return f"Written to {filepath}"


@tool
def run_bash_command(command: str, workspace_dir: str = "./") -> str:
    """Execute a bash command inside the Docker sandbox (in /workspace) and return stdout + stderr."""
    container = get_sandbox(workspace_dir)
    result = container.exec_run(["bash", "-c", command], workdir="/workspace", demux=True)
    stdout = (result.output[0] or b"").decode()
    stderr = (result.output[1] or b"").decode()
    return f"stdout:\n{stdout}\nstderr:\n{stderr}"


@tool
def run_tests(workspace_dir: str = "./") -> str:
    """Run pytest in the Docker sandbox and return the full test output. Use this to check if your changes pass tests."""
    container = get_sandbox(workspace_dir)
    result = container.exec_run(["bash", "-c", "pytest -x -q 2>&1"], workdir="/workspace", demux=False)
    output = (result.output or b"").decode()
    return output[:2000] + "\n[TRUNCATED]" if len(output) > 2000 else output


# Git tools (run inside Docker sandbox)
from tools.git_tools import create_branch, commit_changes, generate_pr_description

tools = [list_files, read_file, search_codebase, write_to_file, run_bash_command, run_tests,
         create_branch, commit_changes, generate_pr_description]

FALLBACK_MODELS = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "groq/llama-3.3-70b-versatile",
    "groq/llama3-8b-8192",
]

_SKIP_ERRORS = ("ResourceExhausted", "RateLimit", "QuotaExceeded", "APIConnectionError", "AuthenticationError", "BadRequestError")

# Transient errors that warrant a retry (rate limits, timeouts, connection issues).
# Auth/bad-request errors are NOT retried — they won't self-heal.
_TRANSIENT_ERROR_NAMES = ("RateLimitError", "ResourceExhausted", "APIConnectionError",
                          "Timeout", "ConnectionError", "ServiceUnavailable", "InternalServerError")
_TRANSIENT_EXCEPTIONS = (Exception,)  # broad catch; filtered by name in _invoke_model


def _model_available(model: str) -> bool:
    if model.startswith("gemini/") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return False
    if model.startswith("groq/") and not os.environ.get("GROQ_API_KEY"):
        return False
    return True


def _make_llm(model: str):
    return ChatLiteLLM(model=model, temperature=0).bind_tools(tools)


def _is_transient(e: Exception) -> bool:
    """Return True if the exception is a transient API error worth retrying."""
    name = type(e).__name__
    msg = str(e).lower()
    return (
        any(t in name for t in _TRANSIENT_ERROR_NAMES) or
        "rate limit" in msg or "timeout" in msg or "connection" in msg or
        "503" in msg or "502" in msg or "529" in msg
    )


def _export_ui_state(state: dict, node: str = "") -> None:
    """Export a serialisable subset of GraphState for the Streamlit UI."""
    cost_summary = _cost_tracker.get_summary()
    ui_state = {
        "iteration_count": state.get("iteration_count", 0),
        "current_node": node or state.get("current_node", "idle"),
        "tests_passed": state.get("tests_passed"),
        "verification_attempts": state.get("verification_attempts", 0),
        "total_cost_usd": cost_summary["total_cost_usd"],
        "budget_exceeded": state.get("budget_exceeded", False),
        "total_tokens": cost_summary["total_tokens"],
        "total_calls": cost_summary["total_calls"],
        "model_breakdown": cost_summary.get("model_breakdown", {}),
        "budget_usd": cost_summary.get("budget_usd", 0.0),
        "last_model_used": state.get("last_model_used", "unknown"),
        "branch_name": state.get("branch_name"),
        "commit_hash": state.get("commit_hash"),
        "messages_count": len(state.get("messages", [])),
        "circuit_status": _circuit_breaker.get_status(),
        "circuit_events": _circuit_events[-20:] if _circuit_events else [],
        "status": "running",
    }
    if state.get("budget_exceeded") or node == "end":
        ui_state["status"] = "completed"
    _state_manager.save_state(ui_state)


def _invoke_model(model: str, msgs: list, max_retries: int, base_delay: float, max_delay: float):
    """Call a single model with exponential backoff on transient errors."""
    @with_retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=2.0,
        retryable_exceptions=_TRANSIENT_EXCEPTIONS,
    )
    def _call():
        try:
            return _make_llm(model).invoke(msgs)
        except Exception as e:
            if _is_transient(e):
                raise  # let with_retry handle it
            raise  # non-transient: re-raise immediately (with_retry will also re-raise)
    return _call()


def planner_node(state: GraphState) -> dict:
    trimmed = []
    for msg in state["messages"][-10:]:
        if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 4000:
            from langchain_core.messages import ToolMessage
            if isinstance(msg, ToolMessage):
                msg = ToolMessage(content=msg.content[:4000] + "\n[TRUNCATED]",
                                  tool_call_id=msg.tool_call_id)
        trimmed.append(msg)
    extra = [NO_WRITE_MSG] if not state.get("writes_performed", False) and state["iteration_count"] > 0 else []
    msgs = [SYSTEM] + extra + trimmed

    for model in FALLBACK_MODELS:
        if not _model_available(model):
            print(f"[SKIP] {model} — no API key set.")
            continue

        # Circuit breaker check
        if not _circuit_breaker.can_call(model):
            event = f"[CIRCUIT OPEN] Skipping {model} (cooldown active)"
            print(event)
            _circuit_events.append(event)
            continue

        print(f"\n--- [NODE] PLANNER | model={model} ---")
        try:
            response = _invoke_model(
                model, msgs,
                max_retries=state.get("_retry_max", 3),
                base_delay=state.get("_retry_delay", 2.0),
                max_delay=30.0,
            )
            _circuit_breaker.record_success(model)

            # Track last successful model for UI
            state["last_model_used"] = model

            # --- Cost tracking ---
            estimated = False
            usage = getattr(response, "usage_metadata", None) or \
                    getattr(response, "response_metadata", {}).get("usage", None)
            if usage:
                input_tokens = (
                    getattr(usage, "prompt_token_count", None) or
                    getattr(usage, "input_tokens", None) or
                    (usage.get("prompt_tokens") if isinstance(usage, dict) else None) or 0
                )
                output_tokens = (
                    getattr(usage, "candidates_token_count", None) or
                    getattr(usage, "output_tokens", None) or
                    (usage.get("completion_tokens") if isinstance(usage, dict) else None) or 0
                )
            else:
                input_tokens = len(msgs) * 500
                output_tokens = len(str(response.content)) // 4
                estimated = True
                print(f"[COST] Token counts unavailable — using estimates (in={input_tokens}, out={output_tokens})")

            _cost_tracker.add_call(model, input_tokens, output_tokens, "planner", estimated)
            total_cost = _cost_tracker.get_total_cost()
            print(f"[COST] ${total_cost:.6f} total | this call: in={input_tokens} out={output_tokens} tokens")

            # Budget check
            if _cost_tracker.check_budget_exceeded():
                print(f"[COST] Budget exceeded (${total_cost:.4f} > ${_cost_tracker.budget_usd}). Halting.")
                budget_msg = SystemMessage(
                    content=f"Budget exceeded (${total_cost:.4f} > ${_cost_tracker.budget_usd}). Halting execution."
                )
                result = {
                    "messages": [response, budget_msg],
                    "iteration_count": state["iteration_count"] + 1,
                    "total_cost_usd": total_cost,
                    "budget_exceeded": True,
                    "tests_passed": False,
                    "current_node": "planner",
                }
                _export_ui_state({**state, **result}, "planner")
                return result

            result = {
                "messages": [response],
                "iteration_count": state["iteration_count"] + 1,
                "total_cost_usd": total_cost,
                "budget_exceeded": False,
                "current_node": "planner",
            }
            _export_ui_state({**state, **result}, "planner")
            return result

        except Exception as e:
            err_name = type(e).__name__
            # Record circuit failure for transient errors; skip for auth/bad-request
            is_permanent = any(t in err_name for t in _SKIP_ERRORS) or \
                           "Missing" in str(e) or "key" in str(e).lower()
            if not is_permanent:
                _circuit_breaker.record_failure(model)
                status = _circuit_breaker.get_status().get(model, {})
                if status.get("state") == "open":
                    event = f"[CIRCUIT OPENED] {model} after {status.get('failures')} failures"
                    _circuit_events.append(event)
            print(f"[FALLBACK] {model} failed: {err_name}. Trying next model...")
            continue

    raise RuntimeError("All models in fallback chain exhausted.")


executor_node = ToolNode(tools)


def _track_tool_calls(state: GraphState) -> dict:
    """Wrap ToolNode to track write_to_file and search_codebase calls."""
    last = state["messages"][-1]
    writes = state.get("writes_performed", False)
    searches = state.get("search_call_count", 0)
    wrote_this_turn = False
    if hasattr(last, "tool_calls"):
        for tc in last.tool_calls:
            if tc["name"] == "write_to_file":
                writes = True
                wrote_this_turn = True
            if tc["name"] == "search_codebase":
                searches += 1
    result = executor_node.invoke(state)
    result["writes_performed"] = writes
    result["search_call_count"] = searches
    result["current_node"] = "executor"
    # Reset tests_passed to None whenever new files are written, so verify_code re-runs
    if wrote_this_turn:
        result["tests_passed"] = None
    _export_ui_state({**state, **result}, "executor")
    return result


def verify_code(state: GraphState) -> dict:
    """Run pytest in the Docker sandbox and update tests_passed / test_output."""
    print("\n--- [NODE] VERIFY ---")
    workspace = state.get("workspace_dir", "./")
    container = get_sandbox(workspace)
    result = container.exec_run(["bash", "-c", "pytest -x -q 2>&1"], workdir="/workspace", demux=False)
    exit_code = result.exit_code
    output = (result.output or b"").decode()
    output = output[:2000] + "\n[TRUNCATED]" if len(output) > 2000 else output

    attempts = state.get("verification_attempts", 0) + 1

    if exit_code == 0:
        print(f"[VERIFY] Tests PASSED (attempt {attempts})")
        result = {
            "tests_passed": True,
            "test_output": "All tests passed.",
            "verification_attempts": attempts,
            "current_node": "verify",
        }
        _export_ui_state({**state, **result}, "verify")
        return result
    else:
        print(f"[VERIFY] Tests FAILED (attempt {attempts}):\n{output[:300]}")
        # Inject failure message so planner sees the errors on next iteration
        error_msg = SystemMessage(content=f"Tests failed. Fix the following errors:\n{output}")
        result = {
            "tests_passed": False,
            "test_output": output,
            "verification_attempts": attempts,
            "messages": [error_msg],
            "current_node": "verify",
        }
        _export_ui_state({**state, **result}, "verify")
        return result


def git_workflow(state: GraphState) -> dict:
    """Auto-create a branch and commit all changes after tests pass."""
    print("\n--- [NODE] GIT WORKFLOW ---")
    workspace = state.get("workspace_dir", "./")
    timestamp = int(time.time())
    branch = f"auto-swe/fix-{timestamp}"

    # Ensure git identity is configured inside container
    from tools.git_tools import _run_in_sandbox
    _run_in_sandbox(
        'git config user.email "agent@auto-swe-agent" && git config user.name "auto-swe-agent"',
        workspace,
    )

    # Check if this is a git repo
    exit_code, _ = _run_in_sandbox("git rev-parse --is-inside-work-tree", workspace)
    if exit_code != 0:
        print("[GIT] Not a git repo — skipping git workflow.")
        result = {"branch_name": None, "commit_hash": None, "current_node": "git_workflow"}
        _export_ui_state({**state, **result}, "git_workflow")
        return result

    # Create branch
    exit_code, out = _run_in_sandbox(f"git checkout -b {branch}", workspace)
    if exit_code != 0:
        print(f"[GIT] Branch creation failed: {out}")
        result = {"branch_name": None, "commit_hash": None, "current_node": "git_workflow"}
        _export_ui_state({**state, **result}, "git_workflow")
        return result
    print(f"[GIT] Created branch: {branch}")

    # Commit all changes
    task_slug = state.get("current_task", "fix")[:50].strip()
    commit_msg = f"auto-swe: {task_slug}"
    _run_in_sandbox("git add -A", workspace)
    exit_code, out = _run_in_sandbox(f'git commit -m "{commit_msg}"', workspace)
    if exit_code != 0:
        print(f"[GIT] Commit failed: {out}")
        result = {"branch_name": branch, "commit_hash": None, "current_node": "git_workflow"}
        _export_ui_state({**state, **result}, "git_workflow")
        return result

    # Parse commit hash
    commit_hash = ""
    for line in out.splitlines():
        if line.startswith("["):
            parts = line.split()
            if len(parts) >= 2:
                commit_hash = parts[1].rstrip("]")
            break
    print(f"[GIT] Committed: {commit_hash} — {commit_msg}")
    result = {"branch_name": branch, "commit_hash": commit_hash, "current_node": "git_workflow"}
    _export_ui_state({**state, **result}, "git_workflow")
    return result


MULTI_FILE_KEYWORDS = ("3 files", "multiple files", "several files", "all files")

NO_WRITE_MSG = SystemMessage(content=(
    "You have not written any files yet. You MUST use write_to_file to implement "
    "the changes before finishing."
))


def route_planner(state: GraphState) -> str:
    """Route after planner: executor → verify → planner cycle."""
    # Budget halt — stop immediately
    if state.get("budget_exceeded"):
        print("[COST] Budget exceeded — routing to end.")
        return "end"

    task = state.get("current_task", "")
    is_multi_file = any(k in task.lower() for k in MULTI_FILE_KEYWORDS) or \
                    state.get("search_call_count", 0) > 3
    limit = 20 if is_multi_file else 15
    if state["iteration_count"] >= limit:
        print(f"[WARNING] Iteration limit ({limit}) reached. Forcing end.")
        return "end"

    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "executor"

    # Hallucination guard: agent claims done but never wrote anything
    if not state.get("writes_performed", False):
        print("[GUARD] No files written yet — forcing back to planner.")
        return "no_write_guard"

    # Agent claims done and has written files — run verification first
    if state.get("tests_passed") is None:
        return "verify"

    return "end"


def route_verify(state: GraphState) -> str:
    """Route after verify_code: git_workflow on pass, planner on failure (max 3), else end."""
    if state.get("tests_passed"):
        return "git_workflow"
    if state.get("verification_attempts", 0) < 3:
        return "planner"
    print("[VERIFY] Max verification attempts reached. Ending.")
    return "end"


# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", _track_tool_calls)
workflow.add_node("verify", verify_code)
workflow.add_node("git_workflow", git_workflow)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", route_planner,
    {"executor": "executor", "end": END, "no_write_guard": "planner", "verify": "verify"}
)
workflow.add_edge("executor", "planner")
workflow.add_conditional_edges(
    "verify", route_verify,
    {"planner": "planner", "git_workflow": "git_workflow", "end": END}
)
workflow.add_edge("git_workflow", END)

app = workflow.compile()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default=None)
    parser.add_argument("--workspace", default="./")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="Max spend in USD (0 = disable tracking)")
    parser.add_argument("--retry-max", type=int, default=3,
                        help="Max retries per model before fallback (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                        help="Base delay in seconds for exponential backoff (default: 2.0)")
    parser.add_argument("--circuit-threshold", type=int, default=5,
                        help="Consecutive failures before circuit opens (default: 5)")
    parser.add_argument("--circuit-timeout", type=int, default=300,
                        help="Seconds before retrying an open circuit (default: 300)")
    args = parser.parse_args()
    task = args.task or input("Enter task: ")
    workspace = os.path.abspath(args.workspace)

    # Reset and configure the module-level tracker for this run
    _cost_tracker.reset()
    _cost_tracker.budget_usd = args.budget

    # Reset and configure the module-level circuit breaker for this run
    _circuit_breaker.reset()
    _circuit_breaker.failure_threshold = args.circuit_threshold
    _circuit_breaker.recovery_timeout = args.circuit_timeout
    _circuit_events.clear()

    print(f"Starting agent for task: {task}\nWorkspace: {workspace}\n"
          f"Budget: {'disabled' if args.budget == 0 else f'${args.budget:.2f}'}\n"
          f"Retry: max={args.retry_max} delay={args.retry_delay}s "
          f"| Circuit: threshold={args.circuit_threshold} timeout={args.circuit_timeout}s\n")

    final_state = app.invoke({
        "messages": [HumanMessage(content=f"Task: {task}")],
        "workspace_dir": workspace,
        "current_task": task,
        "error_logs": "",
        "plan": "",
        "iteration_count": 0,
        "writes_performed": False,
        "search_call_count": 0,
        "tests_passed": None,
        "test_output": None,
        "verification_attempts": 0,
        "branch_name": None,
        "commit_hash": None,
        "total_cost_usd": 0.0,
        "budget_exceeded": False,
        "_retry_max": args.retry_max,
        "_retry_delay": args.retry_delay,
    })
    print("\n=== FINAL ANSWER ===\n")
    content = final_state["messages"][-1].content
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    print(content)

    summary = _cost_tracker.get_summary()
    most_used = summary.get("most_used_model") or "unknown"

    # Print circuit breaker summary
    circuit_status = _circuit_breaker.get_status()
    open_circuits = [m for m, s in circuit_status.items() if s["state"] == "open"]
    if _circuit_events:
        print(f"\n[CIRCUIT EVENTS] ({len(_circuit_events)} total)")
        for ev in _circuit_events[-5:]:
            print(f"  {ev}")
        if open_circuits:
            print(f"  Circuits still open: {', '.join(open_circuits)}")

    # Export final state for UI (mark as completed)
    _export_ui_state({**final_state, "status": "completed"}, "end")

    # Print SUMMARY line (parsed by eval harness)
    print(f"\n[SUMMARY] tests_passed={final_state.get('tests_passed')} | "
          f"verification_attempts={final_state.get('verification_attempts', 0)} | "
          f"branch_name={final_state.get('branch_name')} | "
          f"commit_hash={final_state.get('commit_hash')} | "
          f"total_cost_usd={summary['total_cost_usd']:.6f} | "
          f"total_tokens={summary['total_tokens']} | "
          f"most_used_model={most_used} | "
          f"circuit_events={len(_circuit_events)} | "
          f"circuits_open={len(open_circuits)}")


if __name__ == "__main__":
    main()
