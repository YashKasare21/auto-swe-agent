"""CLI index builder and auto-build helper for auto-swe-agent.

Usage:
    python -m indexing.build_index /path/to/repo [--output path]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from indexing.embedder import CodeEmbedder
from indexing.parser import check_index_staleness, parse_repository
from indexing.vector_store import CodeVectorStore, DEFAULT_INDEX_PATH


def build_index(repo_path: str, output: str = DEFAULT_INDEX_PATH) -> int:
    """Parse, embed, and index a repository. Returns number of chunks indexed."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        print(f"[INDEX] Error: {repo} is not a directory")
        return 0

    print(f"[INDEX] Parsing {repo}...")
    chunks = parse_repository(str(repo))
    if not chunks:
        print("[INDEX] No Python code chunks found.")
        return 0

    print(f"[INDEX] Found {len(chunks)} code chunks. Embedding...")
    embedder = CodeEmbedder()
    embeddings = np.array([embedder.embed_chunk(c) for c in chunks])

    print(f"[INDEX] Building vector index...")
    store = CodeVectorStore(output)
    store.build(chunks, embeddings)
    print(f"[INDEX] Index saved to {output} ({len(chunks)} chunks, {embeddings.shape[1]} dims)")
    return len(chunks)


def ensure_index_built(
    repo_path: str,
    index_path: str = DEFAULT_INDEX_PATH,
    force: bool = False,
) -> bool:
    """Build index if missing or stale. Returns True if index is ready."""
    if force or check_index_staleness(repo_path, index_path):
        store = CodeVectorStore(index_path)
        if store.load():
            if not force and not check_index_staleness(repo_path, index_path):
                return True
        print("[INDEX] Building code index (this may take a minute)...")
        count = build_index(repo_path, index_path)
        return count > 0

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build code index for auto-swe-agent")
    parser.add_argument("repo_path", help="Path to repository to index")
    parser.add_argument("--output", default=DEFAULT_INDEX_PATH, help="Output index path")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    if args.force or check_index_staleness(args.repo_path, args.output):
        count = build_index(args.repo_path, args.output)
        print(f"[INDEX] Done. {count} chunks indexed.")
    else:
        print("[INDEX] Index is up to date.")


if __name__ == "__main__":
    main()
