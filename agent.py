import os
import time
from typing import Annotated, Optional, TypedDict

import docker
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tracking.cost_tracker import CostTracker
from resilience.circuit_breaker import CircuitBreaker
from ui.state_manager import AgentStateManager
from observability.langfuse_client import get_langfuse
from observability.tool_tracing import trace_tool_execution

# ---------------------------------------------------------------------------
# GraphState — the shared state passed between all graph nodes
# ---------------------------------------------------------------------------

MULTI_FILE_KEYWORDS = ("3 files", "multiple files", "several files", "all files")


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    workspace_dir: str
    current_task: str
    error_logs: str
    plan: str
    complexity: Optional[str]
    review_feedback: Optional[str]
    current_agent: Optional[str]
    iteration_count: int
    writes_performed: bool
    search_call_count: int
    semantic_search_call_count: int
    tests_passed: Optional[bool]
    test_output: Optional[str]
    verification_attempts: int
    branch_name: Optional[str]
    commit_hash: Optional[str]
    total_cost_usd: float
    budget_exceeded: bool
    _retry_max: int
    _retry_delay: float
    last_model_used: Optional[str]
    langfuse_trace_id: Optional[str]


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

docker_client = docker.from_env()
_sandbox: docker.models.containers.Container = None
_SANDBOX_LABEL = "auto-swe-agent-sandbox"

# Kept outside GraphState because TypedDict can't hold arbitrary class instances.
_cost_tracker: CostTracker = CostTracker(budget_usd=5.0)
_circuit_breaker: CircuitBreaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300)
_circuit_events: list[str] = []
_state_manager: AgentStateManager = AgentStateManager()

# Per-run counters
_semantic_search_call_count: int = 0
_agent_call_counts: dict[str, int] = {}  # agent_name -> call count for eval tracking
_review_feedbacks: list[str] = []  # LGTM / NEEDS_FIX outcomes for eval tracking

# ---------------------------------------------------------------------------
# Docker sandbox
# ---------------------------------------------------------------------------


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
        raise RuntimeError("[Docker] Health check failed.")
    print("[Docker] Sandbox ready.")
    return _sandbox


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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
    """Run pytest in the Docker sandbox and return the full test output."""
    container = get_sandbox(workspace_dir)
    result = container.exec_run(["bash", "-c", "pytest -x -q 2>&1"], workdir="/workspace", demux=False)
    output = (result.output or b"").decode()
    return output[:2000] + "\n[TRUNCATED]" if len(output) > 2000 else output


from tools.git_tools import create_branch, commit_changes, generate_pr_description
from tools.semantic_search import semantic_search

tools = [
    list_files, read_file, search_codebase, semantic_search, write_to_file,
    run_bash_command, run_tests, create_branch, commit_changes, generate_pr_description,
]

FALLBACK_MODELS = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "groq/llama-3.3-70b-versatile",
    "groq/llama3-8b-8192",
]

# ---------------------------------------------------------------------------
# Shared helpers (used by agent nodes and base)
# ---------------------------------------------------------------------------


def _export_ui_state(state: dict, node: str = "") -> None:
    cost_summary = _cost_tracker.get_summary()
    ui_state = {
        "iteration_count": state.get("iteration_count", 0),
        "current_node": node or state.get("current_node", "idle"),
        "current_agent": state.get("current_agent", "idle"),
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


def _score_run(state: dict) -> None:
    """Score the run in Langfuse based on outcome."""
    trace_id = state.get("langfuse_trace_id")
    langfuse = get_langfuse()
    if not langfuse.is_enabled() or not trace_id:
        return

    tests_passed = state.get("tests_passed")
    score_value = 1.0 if tests_passed else 0.0
    langfuse.score(
        trace_id=trace_id,
        name="tests_passed",
        value=score_value,
        comment=f"verification_attempts={state.get('verification_attempts', 0)} | branch={state.get('branch_name')} | commit={state.get('commit_hash')}",
    )

    if _review_feedbacks:
        lgtm_pct = _review_feedbacks.count("LGTM") / len(_review_feedbacks) * 100
        langfuse.score(
            trace_id=trace_id,
            name="review_quality",
            value=lgtm_pct / 100.0,
            comment=f"lgtm={_review_feedbacks.count('LGTM')} / needs_fix={_review_feedbacks.count('NEEDS_FIX')}",
        )

    if _agent_call_counts:
        total = sum(_agent_call_counts.values())
        if total > 0:
            efficiency = 1.0 - min(_semantic_search_call_count / total, 1.0)
            langfuse.score(
                trace_id=trace_id,
                name="search_efficiency",
                value=efficiency,
                comment=f"semantic_search_calls={_semantic_search_call_count} | total_calls={total}",
            )
    _state_manager.save_state(ui_state)


# ---------------------------------------------------------------------------
# Configure agents base runtime
# ---------------------------------------------------------------------------

from agents.base import configure_runtime

executor_node = ToolNode(tools)
configure_runtime(
    cost_tracker=_cost_tracker,
    circuit_breaker=_circuit_breaker,
    circuit_events=_circuit_events,
    fallback_models=FALLBACK_MODELS,
    tools=tools,
    executor_node=executor_node,
    export_ui_state_fn=_export_ui_state,
)

# ---------------------------------------------------------------------------
# Multi-agent nodes
# ---------------------------------------------------------------------------

from agents.manager import manager_node
from agents.planner import planner_node as multi_planner_node
from agents.coder import coder_node
from agents.reviewer import reviewer_node

# ---------------------------------------------------------------------------
# Shared graph nodes (verify, git, executor)
# ---------------------------------------------------------------------------


def _track_tool_calls(state: GraphState) -> dict:
    """Wrap ToolNode to track tool usage and trace to Langfuse."""
    global _semantic_search_call_count
    last = state["messages"][-1]
    writes = state.get("writes_performed", False)
    searches = state.get("search_call_count", 0)
    semantic_searches = state.get("semantic_search_call_count", 0)
    wrote_this_turn = False
    trace_id = state.get("langfuse_trace_id")

    tool_calls_info = []
    if hasattr(last, "tool_calls"):
        for tc in last.tool_calls:
            tool_calls_info.append((tc["name"], str(tc.get("args", {}))[:200]))
            if tc["name"] == "write_to_file":
                writes = True
                wrote_this_turn = True
            if tc["name"] == "search_codebase":
                searches += 1
            if tc["name"] == "semantic_search":
                semantic_searches += 1
                _semantic_search_call_count += 1

    if trace_id and tool_calls_info:
        span = get_langfuse().span(
            trace_id=trace_id,
            name="tool-execution-batch",
            input={"tool_calls": tool_calls_info},
        )

    result = executor_node.invoke(state)
    result["writes_performed"] = writes
    result["search_call_count"] = searches
    result["semantic_search_call_count"] = semantic_searches
    result["current_node"] = "executor"

    # Trace individual tool results
    if trace_id:
        for tc_name, tc_input in tool_calls_info:
            trace_tool_execution(trace_id, tc_name, tc_input, result)

    if trace_id and tool_calls_info:
        span.update(output={"status": "success", "results_count": len(tool_calls_info)})

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
        return {
            "tests_passed": True,
            "test_output": "All tests passed.",
            "verification_attempts": attempts,
            "current_node": "verify",
        }
    else:
        print(f"[VERIFY] Tests FAILED (attempt {attempts}):\n{output[:300]}")
        error_msg = SystemMessage(content=f"Tests failed. Fix the following errors:\n{output}")
        return {
            "tests_passed": False,
            "test_output": output,
            "verification_attempts": attempts,
            "messages": [error_msg],
            "current_node": "verify",
        }


def git_workflow(state: GraphState) -> dict:
    """Auto-create a branch and commit all changes after tests pass."""
    print("\n--- [NODE] GIT WORKFLOW ---")
    workspace = state.get("workspace_dir", "./")
    timestamp = int(time.time())
    branch = f"auto-swe/fix-{timestamp}"

    from tools.git_tools import _run_in_sandbox
    _run_in_sandbox(
        'git config user.email "agent@auto-swe-agent" && git config user.name "auto-swe-agent"',
        workspace,
    )

    exit_code, _ = _run_in_sandbox("git rev-parse --is-inside-work-tree", workspace)
    if exit_code != 0:
        print("[GIT] Not a git repo — skipping git workflow.")
        return {"branch_name": None, "commit_hash": None, "current_node": "git_workflow"}

    exit_code, out = _run_in_sandbox(f"git checkout -b {branch}", workspace)
    if exit_code != 0:
        print(f"[GIT] Branch creation failed: {out}")
        return {"branch_name": None, "commit_hash": None, "current_node": "git_workflow"}
    print(f"[GIT] Created branch: {branch}")

    task_slug = state.get("current_task", "fix")[:50].strip()
    commit_msg = f"auto-swe: {task_slug}"
    _run_in_sandbox("git add -A", workspace)
    exit_code, out = _run_in_sandbox(f'git commit -m "{commit_msg}"', workspace)
    if exit_code != 0:
        print(f"[GIT] Commit failed: {out}")
        return {"branch_name": branch, "commit_hash": None, "current_node": "git_workflow"}

    commit_hash = ""
    for line in out.splitlines():
        if line.startswith("["):
            parts = line.split()
            if len(parts) >= 2:
                commit_hash = parts[1].rstrip("]")
            break
    print(f"[GIT] Committed: {commit_hash} — {commit_msg}")
    return {"branch_name": branch, "commit_hash": commit_hash, "current_node": "git_workflow"}


# ---------------------------------------------------------------------------
# Routing functions (multi-agent)
# ---------------------------------------------------------------------------


def route_manager(state: GraphState) -> str:
    target = "end" if state.get("error_logs") else "planner"
    log_routing(state, "manager", target)
    return target


def route_planner(state: GraphState) -> str:
    target = "end" if state.get("error_logs") else "coder"
    log_routing(state, "planner", target)
    return target


def route_coder(state: GraphState) -> str:
    if state.get("error_logs"):
        log_routing(state, "coder", "end")
        return "end"
    if state.get("budget_exceeded"):
        print("[COST] Budget exceeded — routing to end.")
        log_routing(state, "coder", "end")
        return "end"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        log_routing(state, "coder", "executor")
        return "executor"
    task = state.get("current_task", "")
    is_multi_file = any(k in task.lower() for k in MULTI_FILE_KEYWORDS) or \
                    state.get("search_call_count", 0) > 3
    limit = 20 if is_multi_file else 15
    if state["iteration_count"] >= limit:
        print(f"[WARNING] Iteration limit ({limit}) reached. Forcing end.")
        log_routing(state, "coder", "end")
        return "end"
    if not state.get("writes_performed", False):
        print("[GUARD] No files written yet — forcing back to coder.")
        log_routing(state, "coder", "coder")
        return "coder"
    log_routing(state, "coder", "verify")
    return "verify"


def route_verify(state: GraphState) -> str:
    if state.get("tests_passed"):
        log_routing(state, "verify", "reviewer")
        return "reviewer"
    if state.get("verification_attempts", 0) < 3:
        log_routing(state, "verify", "coder")
        return "coder"
    print("[VERIFY] Max verification attempts reached. Ending.")
    log_routing(state, "verify", "end")
    return "end"


def route_reviewer(state: GraphState) -> str:
    review = state.get("review_feedback", state.get("messages", [{}])[-1].content if state.get("messages") else "").upper()
    if "LGTM" in review:
        global _review_feedbacks
        _review_feedbacks.append("LGTM")
        log_routing(state, "reviewer", "git_workflow")
        return "git_workflow"
    _review_feedbacks.append("NEEDS_FIX")
    log_routing(state, "reviewer", "coder")
    return "coder"


def route_git(state: GraphState) -> str:
    log_routing(state, "git_workflow", "end")
    return "end"


def log_routing(state: GraphState, source: str, target: str) -> None:
    """Trace routing decisions to Langfuse."""
    trace_id = state.get("langfuse_trace_id")
    if trace_id:
        langfuse = get_langfuse()
        if langfuse.is_enabled():
            span = langfuse.span(
                trace_id=trace_id,
                name=f"routing-{source}->{target}",
                input={
                    "source": source,
                    "target": target,
                    "iteration": state.get("iteration_count"),
                    "tests_passed": state.get("tests_passed"),
                    "writes_performed": state.get("writes_performed"),
                    "budget_exceeded": state.get("budget_exceeded"),
                },
            )
            span.update(output={"status": "routed"})


# ---------------------------------------------------------------------------
# Single-agent routing (backward compatible)
# ---------------------------------------------------------------------------


def route_planner_single(state: GraphState) -> str:
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
    if not state.get("writes_performed", False):
        print("[GUARD] No files written yet — forcing back to planner.")
        return "planner"
    if state.get("tests_passed") is None:
        return "verify"
    return "end"


def route_verify_single(state: GraphState) -> str:
    if state.get("tests_passed"):
        return "git_workflow"
    if state.get("verification_attempts", 0) < 3:
        return "planner"
    print("[VERIFY] Max verification attempts reached. Ending.")
    return "end"


# ---------------------------------------------------------------------------
# Single-agent node (old planner logic, kept for backward compat)
# ---------------------------------------------------------------------------

NO_WRITE_MSG = SystemMessage(content=(
    "You have not written any files yet. You MUST use write_to_file to implement "
    "the changes before finishing."
))

SINGLE_AGENT_SYSTEM = """You are an autonomous coding agent that fixes bugs and implements features. \
You have access to the following tools in two categories:

=== SEARCH TOOLS ===
1. search_codebase(keyword, directory) — Exact text matching. \
Use for finding specific strings, variable names, function references.
2. semantic_search(query, k=5) — Semantic (meaning-based) search. \
Use for finding code by concept or functionality. \
Example: 'find where user authentication is handled' → semantic_search. \
'find all occurrences of password' → search_codebase.

=== FILE / EXECUTION TOOLS ===
3. list_files(directory) — Show directory tree.
4. read_file(filepath) — Read a file (truncated at 2000 lines).
5. write_to_file(filepath, content) — Write/replace a file.
6. run_bash_command(command, workspace_dir) — Execute bash inside Docker sandbox.
7. run_tests(workspace_dir) — Run pytest inside Docker sandbox.

=== GIT TOOLS ===
8. create_branch(branch_name) — Create a new git branch.
9. commit_changes(message) — Stage and commit all changes.
10. generate_pr_description() — Generate a PR description from the diff.

=== RULES ===
- Always run tests (run_tests) after writing code to verify correctness.
- If tests fail, read the error output and fix the code.
- You MUST use write_to_file at least once before declaring the task done.
- Prefer semantic_search for understanding code structure and finding logic; \
use search_codebase only for exact string lookups.
- You have up to 15 iterations (20 for multi-file tasks) to complete the task."""


def _invoke_model(model: str, msgs: list, max_retries: int, base_delay: float, max_delay: float):
    from resilience.retry import with_retry
    from agents.base import _is_transient
    from langchain_community.chat_models import ChatLiteLLM

    llm = ChatLiteLLM(model=model, temperature=0).bind_tools(tools)

    @with_retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=2.0,
        retryable_exceptions=(Exception,),
    )
    def _call():
        try:
            return llm.invoke(msgs)
        except Exception as e:
            if _is_transient(e):
                raise
            raise
    return _call()


def planner_node_single(state: GraphState) -> dict:
    trimmed = []
    for msg in state["messages"][-10:]:
        if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 4000:
            from langchain_core.messages import ToolMessage
            if isinstance(msg, ToolMessage):
                msg = ToolMessage(content=msg.content[:4000] + "\n[TRUNCATED]",
                                  tool_call_id=msg.tool_call_id)
        trimmed.append(msg)
    extra = [NO_WRITE_MSG] if not state.get("writes_performed", False) and state["iteration_count"] > 0 else []
    msgs = [SystemMessage(content=SINGLE_AGENT_SYSTEM)] + extra + trimmed

    for model in FALLBACK_MODELS:
        if model.startswith("gemini/") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
            print(f"[SKIP] {model} — no API key set.")
            continue
        if model.startswith("groq/") and not os.environ.get("GROQ_API_KEY"):
            print(f"[SKIP] {model} — no API key set.")
            continue

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
            state["last_model_used"] = model

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
            is_permanent = any(t in err_name for t in (
                "ResourceExhausted", "RateLimit", "QuotaExceeded",
                "APIConnectionError", "AuthenticationError", "BadRequestError",
            )) or "Missing" in str(e) or "key" in str(e).lower()
            if not is_permanent:
                _circuit_breaker.record_failure(model)
                status = _circuit_breaker.get_status().get(model, {})
                if status.get("state") == "open":
                    event = f"[CIRCUIT OPENED] {model} after {status.get('failures')} failures"
                    _circuit_events.append(event)
            print(f"[FALLBACK] {model} failed: {err_name}. Trying next model...")
            continue

    raise RuntimeError("All models in fallback chain exhausted.")


# ---------------------------------------------------------------------------
# Build graph (architecture selected by --single-agent flag)
# ---------------------------------------------------------------------------

_single_agent_mode = False
app = None


def _build_multi_agent_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("manager", manager_node)
    workflow.add_node("planner", multi_planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("executor", _track_tool_calls)
    workflow.add_node("verify", verify_code)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("git_workflow", git_workflow)

    workflow.set_entry_point("manager")

    workflow.add_conditional_edges("manager", route_manager, {
        "planner": "planner", "end": END,
    })
    workflow.add_conditional_edges("planner", route_planner, {
        "coder": "coder", "end": END,
    })
    workflow.add_conditional_edges("coder", route_coder, {
        "executor": "executor", "verify": "verify", "end": END,
    })
    workflow.add_edge("executor", "coder")
    workflow.add_conditional_edges("verify", route_verify, {
        "reviewer": "reviewer", "coder": "coder", "end": END,
    })
    workflow.add_conditional_edges("reviewer", route_reviewer, {
        "git_workflow": "git_workflow", "coder": "coder",
    })
    workflow.add_conditional_edges("git_workflow", route_git, {
        "end": END,
    })
    return workflow.compile()


def _build_single_agent_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("planner", planner_node_single)
    workflow.add_node("executor", _track_tool_calls)
    workflow.add_node("verify", verify_code)
    workflow.add_node("git_workflow", git_workflow)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", route_planner_single, {
        "executor": "executor", "end": END, "planner": "planner", "verify": "verify",
    })
    workflow.add_edge("executor", "planner")
    workflow.add_conditional_edges("verify", route_verify_single, {
        "planner": "planner", "git_workflow": "git_workflow", "end": END,
    })
    workflow.add_edge("git_workflow", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    global _single_agent_mode, app, _semantic_search_call_count, _agent_call_counts, _review_feedbacks

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default=None)
    parser.add_argument("--workspace", default="./")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--retry-max", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--circuit-threshold", type=int, default=5)
    parser.add_argument("--circuit-timeout", type=int, default=300)
    parser.add_argument("--single-agent", action="store_true",
                        help="Use single-agent mode (backward-compatible planner-only)")
    args = parser.parse_args()
    task = args.task or input("Enter task: ")
    workspace = os.path.abspath(args.workspace)

    _single_agent_mode = args.single_agent
    _semantic_search_call_count = 0
    _agent_call_counts = {}
    _review_feedbacks = []

    from indexing.build_index import ensure_index_built
    ensure_index_built(workspace)

    _cost_tracker.reset()
    _cost_tracker.budget_usd = args.budget
    _circuit_breaker.reset()
    _circuit_breaker.failure_threshold = args.circuit_threshold
    _circuit_breaker.recovery_timeout = args.circuit_timeout
    _circuit_events.clear()

    mode = "single-agent" if _single_agent_mode else "multi-agent"
    print(f"Starting agent for task: {task}\nWorkspace: {workspace}\n"
          f"Mode: {mode}\n"
          f"Budget: {'disabled' if args.budget == 0 else f'${args.budget:.2f}'}\n"
          f"Retry: max={args.retry_max} delay={args.retry_delay}s "
          f"| Circuit: threshold={args.circuit_threshold} timeout={args.circuit_timeout}s\n")

    # Build the appropriate graph
    if _single_agent_mode:
        app = _build_single_agent_graph()
    else:
        app = _build_multi_agent_graph()

    initial_state: GraphState = {
        "messages": [HumanMessage(content=f"Task: {task}")],
        "workspace_dir": workspace,
        "current_task": task,
        "error_logs": "",
        "plan": "",
        "complexity": None,
        "review_feedback": None,
        "current_agent": None,
        "iteration_count": 0,
        "writes_performed": False,
        "search_call_count": 0,
        "semantic_search_call_count": 0,
        "tests_passed": None,
        "test_output": None,
        "verification_attempts": 0,
        "branch_name": None,
        "commit_hash": None,
        "total_cost_usd": 0.0,
        "budget_exceeded": False,
        "_retry_max": args.retry_max,
        "_retry_delay": args.retry_delay,
        "last_model_used": None,
        "langfuse_trace_id": None,
    }

    final_state = app.invoke(initial_state)
    print("\n=== FINAL ANSWER ===\n")
    content = final_state["messages"][-1].content
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    print(content)

    summary = _cost_tracker.get_summary()
    most_used = summary.get("most_used_model") or "unknown"

    circuit_status = _circuit_breaker.get_status()
    open_circuits = [m for m, s in circuit_status.items() if s["state"] == "open"]
    if _circuit_events:
        print(f"\n[CIRCUIT EVENTS] ({len(_circuit_events)} total)")
        for ev in _circuit_events[-5:]:
            print(f"  {ev}")
        if open_circuits:
            print(f"  Circuits still open: {', '.join(open_circuits)}")

    _export_ui_state({**final_state, "status": "completed"}, "end")

    lgtm_count = _review_feedbacks.count("LGTM")
    needs_fix_count = _review_feedbacks.count("NEEDS_FIX")

    print(f"\n[SUMMARY] tests_passed={final_state.get('tests_passed')} | "
          f"verification_attempts={final_state.get('verification_attempts', 0)} | "
          f"branch_name={final_state.get('branch_name')} | "
          f"commit_hash={final_state.get('commit_hash')} | "
          f"total_cost_usd={summary['total_cost_usd']:.6f} | "
          f"total_tokens={summary['total_tokens']} | "
          f"most_used_model={most_used} | "
          f"circuit_events={len(_circuit_events)} | "
          f"circuits_open={len(open_circuits)} | "
          f"semantic_search_calls={_semantic_search_call_count} | "
          f"lgtm={lgtm_count} | "
          f"needs_fix={needs_fix_count}")

    # Langfuse observability: score and flush
    _score_run(final_state)

    trace_id = final_state.get("langfuse_trace_id")
    if trace_id:
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(f"\n[LANGFUSE] Trace: {host}/trace/{trace_id}")

    get_langfuse().flush()


if __name__ == "__main__":
    main()
