import os
from typing import Annotated, Optional, TypedDict

import docker
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

docker_client = docker.from_env()
_sandbox: docker.models.containers.Container = None
_SANDBOX_LABEL = "auto-swe-agent-sandbox"


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


tools = [list_files, read_file, search_codebase, write_to_file, run_bash_command, run_tests]

FALLBACK_MODELS = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "groq/llama-3.3-70b-versatile",
    "groq/llama3-8b-8192",
]

_SKIP_ERRORS = ("ResourceExhausted", "RateLimit", "QuotaExceeded", "APIConnectionError", "AuthenticationError", "BadRequestError")


def _model_available(model: str) -> bool:
    if model.startswith("gemini/") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return False
    if model.startswith("groq/") and not os.environ.get("GROQ_API_KEY"):
        return False
    return True


def _make_llm(model: str):
    return ChatLiteLLM(model=model, temperature=0).bind_tools(tools)


SYSTEM = SystemMessage(content=(
    "You are an autonomous software engineer. Use list_files, read_file, and search_codebase "
    "to explore the codebase, then use write_to_file to implement fixes, and run_bash_command "
    "to run tests and verify your changes. You can also call run_tests at any time to check "
    "test status. Do not just plan — actually implement and verify the fix. "
    "When tests pass, state that the task is complete."
))


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    workspace_dir: str
    current_task: str
    error_logs: str
    plan: str
    iteration_count: int
    writes_performed: bool
    search_call_count: int
    # --- verification loop fields ---
    tests_passed: Optional[bool]       # None=not yet run, True=pass, False=fail
    test_output: Optional[str]         # stdout/stderr from last pytest run
    verification_attempts: int         # number of fix retries so far


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
        print(f"\n--- [NODE] PLANNER | model={model} ---")
        try:
            response = _make_llm(model).invoke(msgs)
            return {"messages": [response], "iteration_count": state["iteration_count"] + 1}
        except Exception as e:
            if any(t in type(e).__name__ for t in _SKIP_ERRORS) or "Missing" in str(e) or "key" in str(e).lower():
                print(f"[FALLBACK] {model} failed: {type(e).__name__}. Trying next model...")
                continue
            raise
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
    # Reset tests_passed to None whenever new files are written, so verify_code re-runs
    if wrote_this_turn:
        result["tests_passed"] = None
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
        }
    else:
        print(f"[VERIFY] Tests FAILED (attempt {attempts}):\n{output[:300]}")
        # Inject failure message so planner sees the errors on next iteration
        error_msg = SystemMessage(content=f"Tests failed. Fix the following errors:\n{output}")
        return {
            "tests_passed": False,
            "test_output": output,
            "verification_attempts": attempts,
            "messages": [error_msg],
        }


MULTI_FILE_KEYWORDS = ("3 files", "multiple files", "several files", "all files")

NO_WRITE_MSG = SystemMessage(content=(
    "You have not written any files yet. You MUST use write_to_file to implement "
    "the changes before finishing."
))


def route_planner(state: GraphState) -> str:
    """Route after planner: executor → verify → planner cycle."""
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
    """Route after verify_code: back to planner on failure (up to 3 attempts), else end."""
    if state.get("tests_passed"):
        return "end"
    if state.get("verification_attempts", 0) < 3:
        return "planner"
    print("[VERIFY] Max verification attempts reached. Ending.")
    return "end"


# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", _track_tool_calls)
workflow.add_node("verify", verify_code)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", route_planner,
    {"executor": "executor", "end": END, "no_write_guard": "planner", "verify": "verify"}
)
workflow.add_edge("executor", "planner")
workflow.add_conditional_edges(
    "verify", route_verify,
    {"planner": "planner", "end": END}
)

app = workflow.compile()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default=None)
    parser.add_argument("--workspace", default="./")
    args = parser.parse_args()
    task = args.task or input("Enter task: ")
    workspace = os.path.abspath(args.workspace)
    print(f"Starting agent for task: {task}\nWorkspace: {workspace}\n")
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
    })
    print("\n=== FINAL ANSWER ===\n")
    content = final_state["messages"][-1].content
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    print(content)
    # Print verification summary
    print(f"\n[SUMMARY] tests_passed={final_state.get('tests_passed')} | "
          f"verification_attempts={final_state.get('verification_attempts', 0)}")


if __name__ == "__main__":
    main()
