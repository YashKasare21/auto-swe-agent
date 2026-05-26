"""Code repository indexing and semantic search for auto-swe-agent."""

from indexing.build_index import ensure_index_built
from indexing.embedder import CodeEmbedder
from indexing.parser import (
    CodeChunk,
    check_index_staleness,
    parse_file,
    parse_repository,
)
from indexing.vector_store import CodeVectorStore

__all__ = [
    "CodeChunk",
    "parse_file",
    "parse_repository",
    "check_index_staleness",
    "CodeEmbedder",
    "CodeVectorStore",
    "ensure_index_built",
]
