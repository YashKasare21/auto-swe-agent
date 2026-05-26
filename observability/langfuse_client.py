import os
from typing import Any, Dict, Optional


class LangfuseSpan:
    def __init__(self, span=None):
        self._span = span

    def update(self, output: Any = None, **kwargs) -> None:
        if self._span is not None:
            try:
                kwargs["output"] = output
                self._span.update(**kwargs)
            except Exception:
                pass


class LangfuseClient:
    def __init__(self):
        self.enabled = bool(
            os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
            and os.getenv("LANGFUSE_HOST")
        )
        self._langfuse = None
        self._module = None
        if self.enabled:
            try:
                from langfuse import Langfuse as _Langfuse

                self._module = _Langfuse
                self._langfuse = _Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
            except ImportError:
                self.enabled = False
                print(
                    "[Langfuse] langfuse package not installed. Install with: pip install langfuse"
                )
        if not self.enabled:
            print(
                "[Langfuse] Not configured. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST"
            )

    def is_enabled(self) -> bool:
        return self.enabled and self._langfuse is not None

    def create_trace(self, name: str, metadata: Optional[Dict] = None):
        if not self.is_enabled():
            return None
        try:
            return self._langfuse.trace(name=name, metadata=metadata or {})
        except Exception as e:
            print(f"[Langfuse] Failed to create trace: {e}")
            return None

    def span(self, trace_id: str, name: str, input: Any = None, output: Any = None):
        if not self.is_enabled():
            return LangfuseSpan()
        try:
            span = self._langfuse.span(
                trace_id=trace_id,
                name=name,
                input=input,
                output=output,
            )
            return LangfuseSpan(span)
        except Exception as e:
            print(f"[Langfuse] Failed to create span: {e}")
            return LangfuseSpan()

    def generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: Optional[Dict] = None,
    ):
        if not self.is_enabled():
            return None
        try:
            return self._langfuse.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                input=input,
                output=output,
                usage=usage,
            )
        except Exception as e:
            print(f"[Langfuse] Failed to create generation: {e}")
            return None

    def score(self, trace_id: str, name: str, value: float, comment: str = ""):
        if not self.is_enabled():
            return None
        try:
            return self._langfuse.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            print(f"[Langfuse] Failed to score: {e}")
            return None

    def flush(self):
        if self.is_enabled():
            try:
                self._langfuse.flush()
            except Exception as e:
                print(f"[Langfuse] Flush failed: {e}")


_langfuse_client = LangfuseClient()


def get_langfuse() -> LangfuseClient:
    return _langfuse_client
