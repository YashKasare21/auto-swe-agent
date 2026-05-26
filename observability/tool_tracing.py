from functools import wraps
from typing import Any, Callable, Optional

from observability.langfuse_client import get_langfuse


def trace_tool(tool_name: str) -> Callable:
    """Decorator for tracing individual tool function calls.

    Usage:
        @trace_tool("write_to_file")
        def my_tool(filepath: str, content: str) -> str:
            ...

    The decorator expects a `_trace_id` keyword argument that it will pop
    before calling the wrapped function (so it doesn't interfere with the
    tool's signature). For LangChain @tool decorated functions, this is
    applied AFTER @tool but the caller must pass _trace_id explicitly.

    Note: For LangChain ToolNode execution, tracing happens automatically
    via trace_tool_execution() in the executor node wrapper.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            trace_id = kwargs.pop("_trace_id", None)
            langfuse = get_langfuse()
            span = None
            if langfuse.is_enabled() and trace_id:
                span = langfuse.span(
                    trace_id=trace_id,
                    name=f"tool-{tool_name}",
                    input={"args": str(args), "kwargs": str(kwargs)},
                )
            try:
                result = func(*args, **kwargs)
                if span is not None:
                    span.update(output={"status": "success", "result_length": len(str(result))})
                return result
            except Exception as e:
                if span is not None:
                    span.update(output={"status": "error", "error": str(e)})
                raise
        return wrapper
    return decorator


def trace_tool_execution(
    trace_id: str,
    tool_name: str,
    input_repr: str,
    result: Any,
    error: Optional[str] = None,
) -> None:
    """Log a tool execution to Langfuse from the executor node.

    This is called from _track_tool_calls() in agent.py for each tool
    invocation dispatched through ToolNode. It does not modify the tool
    function itself.
    """
    langfuse = get_langfuse()
    if not langfuse.is_enabled() or not trace_id:
        return
    span = langfuse.span(
        trace_id=trace_id,
        name=f"tool-{tool_name}",
        input={"args": input_repr},
    )
    if error:
        span.update(output={"status": "error", "error": error})
    else:
        span.update(output={"status": "success", "result_length": len(str(result))})
