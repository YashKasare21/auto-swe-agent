import os
import time
from typing import Any, Callable, Optional

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import SystemMessage
from observability.langfuse_client import get_langfuse

_SKIP_ERRORS = (
    "ResourceExhausted", "RateLimit", "QuotaExceeded",
    "APIConnectionError", "AuthenticationError", "BadRequestError",
)
_TRANSIENT_ERROR_NAMES = (
    "RateLimitError", "ResourceExhausted", "APIConnectionError",
    "Timeout", "ConnectionError", "ServiceUnavailable", "InternalServerError",
)
_TRANSIENT_EXCEPTIONS = (Exception,)


class AgentRuntime:
    cost_tracker: Any = None
    circuit_breaker: Any = None
    circuit_events: list = []
    fallback_models: list[str] = []
    tools: list = []
    executor_node: Any = None
    export_ui_state_fn: Optional[Callable] = None


_runtime = AgentRuntime()


def configure_runtime(
    cost_tracker,
    circuit_breaker,
    circuit_events: list,
    fallback_models: list[str],
    tools: list,
    executor_node,
    export_ui_state_fn: Optional[Callable] = None,
) -> None:
    _runtime.cost_tracker = cost_tracker
    _runtime.circuit_breaker = circuit_breaker
    _runtime.circuit_events = circuit_events
    _runtime.fallback_models = fallback_models
    _runtime.tools = tools
    _runtime.executor_node = executor_node
    _runtime.export_ui_state_fn = export_ui_state_fn


def get_runtime() -> AgentRuntime:
    return _runtime


def _model_available(model: str) -> bool:
    if model.startswith("gemini/") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return False
    if model.startswith("groq/") and not os.environ.get("GROQ_API_KEY"):
        return False
    return True


def _make_llm(model: str, tools_list: list):
    return ChatLiteLLM(model=model, temperature=0).bind_tools(tools_list)


def _is_transient(e: Exception) -> bool:
    name = type(e).__name__
    msg = str(e).lower()
    return (
        any(t in name for t in _TRANSIENT_ERROR_NAMES)
        or "rate limit" in msg or "timeout" in msg or "connection" in msg
        or "503" in msg or "502" in msg or "529" in msg
    )


def _call_with_retry(model: str, msgs: list, max_retries: int, base_delay: float):
    for attempt in range(max_retries + 1):
        try:
            llm = _make_llm(model, _runtime.tools)
            return llm.invoke(msgs)
        except Exception as e:
            if attempt >= max_retries or not _is_transient(e):
                raise
            delay = min(base_delay * (2.0 ** attempt), 30.0)
            print(f"[RETRY] {model} attempt {attempt + 1}/{max_retries} failed ({type(e).__name__}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


def _extract_usage(response) -> Optional[dict]:
    usage = getattr(response, "usage_metadata", None) or \
            getattr(response, "response_metadata", {}).get("usage", None)
    if usage is None:
        return None
    input_tokens = (
        getattr(usage, "prompt_token_count", None)
        or getattr(usage, "input_tokens", None)
        or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
        or 0
    )
    output_tokens = (
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "output_tokens", None)
        or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
        or 0
    )
    return {"input": input_tokens, "output": output_tokens, "unit": "TOKENS"}


def invoke_agent(
    system_prompt: str,
    state: dict,
    node_name: str,
    *,
    extra_messages: Optional[list] = None,
    context_window: int = 10,
) -> dict:
    cost_tracker = _runtime.cost_tracker
    circuit_breaker = _runtime.circuit_breaker
    circuit_events = _runtime.circuit_events
    export_ui_state_fn = _runtime.export_ui_state_fn

    langfuse = get_langfuse()
    trace_id = state.get("langfuse_trace_id")

    if langfuse.is_enabled() and not trace_id:
        trace = langfuse.create_trace(
            name=f"auto-swe-agent",
            metadata={
                "task": state.get("current_task", "unknown")[:200],
                "workspace": state.get("workspace_dir", "unknown"),
                "mode": "multi-agent",
            },
        )
        if trace is not None and hasattr(trace, "id"):
            trace_id = trace.id

    trimmed = []
    for msg in state["messages"][-context_window:]:
        if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 4000:
            from langchain_core.messages import ToolMessage
            if isinstance(msg, ToolMessage):
                msg = ToolMessage(
                    content=msg.content[:4000] + "\n[TRUNCATED]",
                    tool_call_id=msg.tool_call_id,
                )
        trimmed.append(msg)

    msgs = [SystemMessage(content=system_prompt)] + (extra_messages or []) + trimmed
    last_input = trimmed[-1].content if trimmed else ""

    agent_span = None
    if langfuse.is_enabled() and trace_id:
        agent_span = langfuse.span(
            trace_id=trace_id,
            name=f"agent-{node_name}",
            input={
                "messages_count": len(msgs),
                "context_window": context_window,
                "last_input_preview": str(last_input)[:300],
            },
        )

    for model in _runtime.fallback_models:
        if not _model_available(model):
            print(f"[SKIP] {model} — no API key set.")
            continue

        if not circuit_breaker.can_call(model):
            event = f"[CIRCUIT OPEN] Skipping {model} (cooldown active)"
            print(event)
            circuit_events.append(event)
            continue

        print(f"\n--- [NODE] {node_name.upper()} | model={model} ---")

        try:
            response = _call_with_retry(
                model, msgs,
                max_retries=state.get("_retry_max", 3),
                base_delay=state.get("_retry_delay", 2.0),
            )
            circuit_breaker.record_success(model)

            usage_dict = _extract_usage(response)
            if usage_dict:
                input_tokens = usage_dict["input"]
                output_tokens = usage_dict["output"]
                estimated = False
            else:
                input_tokens = len(msgs) * 500
                output_tokens = len(str(response.content)) // 4
                estimated = True
                print(f"[COST] Token counts unavailable — using estimates (in={input_tokens}, out={output_tokens})")

            # Langfuse generation trace
            if langfuse.is_enabled() and trace_id:
                gen_params = {
                    "trace_id": trace_id,
                    "name": f"llm-{node_name}",
                    "model": model,
                    "input": str(last_input)[:500],
                    "output": str(response.content)[:1000] if hasattr(response, "content") else str(response)[:1000],
                }
                if usage_dict:
                    gen_params["usage"] = usage_dict
                langfuse.generation(**gen_params)

            cost_tracker.add_call(model, input_tokens, output_tokens, node_name, estimated)
            total_cost = cost_tracker.get_total_cost()
            print(f"[COST] ${total_cost:.6f} total | this call: in={input_tokens} out={output_tokens} tokens")

            if agent_span is not None:
                agent_span.update(output={"status": "success", "model_used": model})

            if cost_tracker.check_budget_exceeded():
                print(f"[COST] Budget exceeded (${total_cost:.4f} > ${cost_tracker.budget_usd}).")
                budget_msg = SystemMessage(
                    content=f"Budget exceeded (${total_cost:.4f} > ${cost_tracker.budget_usd}). Halting."
                )
                result = {
                    "messages": [response, budget_msg],
                    "iteration_count": state["iteration_count"] + 1,
                    "total_cost_usd": total_cost,
                    "budget_exceeded": True,
                    "tests_passed": False,
                    "current_node": node_name,
                    "current_agent": node_name,
                }
                if trace_id:
                    result["langfuse_trace_id"] = trace_id
                if export_ui_state_fn:
                    export_ui_state_fn({**state, **result}, node_name)
                return result

            result = {
                "messages": [response],
                "iteration_count": state["iteration_count"] + 1,
                "total_cost_usd": total_cost,
                "budget_exceeded": False,
                "current_node": node_name,
                "current_agent": node_name,
            }
            if trace_id:
                result["langfuse_trace_id"] = trace_id
            if export_ui_state_fn:
                export_ui_state_fn({**state, **result}, node_name)
            return result

        except Exception as e:
            err_name = type(e).__name__
            is_permanent = (
                any(t in err_name for t in _SKIP_ERRORS)
                or "Missing" in str(e) or "key" in str(e).lower()
            )
            if not is_permanent:
                circuit_breaker.record_failure(model)
                status = circuit_breaker.get_status().get(model, {})
                if status.get("state") == "open":
                    event = f"[CIRCUIT OPENED] {model} after {status.get('failures')} failures"
                    circuit_events.append(event)
            print(f"[FALLBACK] {model} failed: {err_name}. Trying next model...")
            if agent_span is not None:
                agent_span.update(output={"status": "fallback", "error": err_name, "model": model})
            continue

    if agent_span is not None:
        agent_span.update(output={"status": "error", "error": "All models exhausted"})
    raise RuntimeError("All models in fallback chain exhausted.")
