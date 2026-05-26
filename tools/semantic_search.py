"""Semantic code search tool for auto-swe-agent.

Provides a LangChain tool that searches the codebase by meaning
using the pre-built vector index.
"""
from __future__ import annotations

from langchain_core.tools import tool

from indexing.embedder import CodeEmbedder
from indexing.vector_store import CodeVectorStore, DEFAULT_INDEX_PATH

# Module-level cache so the index is loaded once per agent run
_store: CodeVectorStore | None = None
_embedder: CodeEmbedder | None = None


def _ensure_store() -> CodeVectorStore:
    global _store
    if _store is None:
        _store = CodeVectorStore(DEFAULT_INDEX_PATH)
        _store.load()
    return _store


def _ensure_embedder() -> CodeEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = CodeEmbedder()
    return _embedder


@tool
def semantic_search(query: str, k: int = 5) -> str:
    """
    Search the codebase for code that is semantically related to your query.
    Use this when you need to find code by meaning, concept, or functionality
    — not just by exact text matches.

    Good examples:
      "find where user authentication is handled"
      "where is the database connection configured"
      "how are errors logged in this codebase"
      "find the function that validates email addresses"

    Bad examples (use search_codebase instead):
      "find all occurrences of 'password'"
      "search for 'TODO' comments"

    Args:
        query: Natural language description of what you are looking for.
        k: Number of results to return (default 5, max 15).

    Returns:
        Formatted text with the top-k matching code chunks, including
        file path, line number, signature, docstring, and a preview.
    """
    store = _ensure_store()
    if not store.metadata:
        return (
            "[SEMANTIC SEARCH] No index found. "
            "The code index has not been built yet — try running the agent "
            "first to trigger auto-indexing, or use search_codebase as a fallback."
        )

    embedder = _ensure_embedder()
    query_vec = embedder.embed([query])[0]
    results = store.search(query_vec, min(k, 15))

    if not results:
        return "[SEMANTIC SEARCH] No matches found."

    output = []
    for chunk, score in results:
        output.append(
            f"=== {chunk.file_path}:{chunk.start_line} "
            f"(score: {score:.3f}, type: {chunk.chunk_type}) ==="
        )
        output.append(f"  {chunk.signature}")
        if chunk.docstring:
            doc = chunk.docstring.strip()[:200]
            output.append(f"  Docstring: {doc}")
        body = chunk.body_preview[:250].strip()
        if body:
            output.append(f"  Preview:\n{body}")
        output.append("")

    return "\n".join(output)
