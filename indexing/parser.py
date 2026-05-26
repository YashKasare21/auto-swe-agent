"""AST-based Python code parser.

Extracts function/class/module chunks with docstrings, signatures, and
line ranges from Python files. Uses stdlib ``ast`` — no external parser needed.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

# Directories to skip when walking repositories
IGNORE_DIRS: Set[str] = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    "node_modules",
    ".next",
    ".mypy_cache",
    ".tox",
    "build",
    "dist",
    "*.egg-info",
    ".eggs",
    ".git",
    "htmlcov",
}


@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str  # "function", "class", "module"
    name: str
    signature: str
    docstring: str
    start_line: int
    end_line: int
    body_preview: str  # first 500 chars of body
    full_text: str = ""  # complete source of the chunk


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from an AST node body, if present."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return ast.get_docstring(node) or ""
    return ""


def _chunk_text(file_text: str, node: ast.AST, start: int, end: int) -> tuple[str, str]:
    """Return (body_preview, full_text) for a node's line range."""
    lines = file_text.splitlines(keepends=True)
    segment = "".join(lines[start - 1 : end])
    preview = segment[:500]
    return preview, segment


def parse_file(filepath: str) -> List[CodeChunk]:
    """Parse a single Python file into CodeChunks.

    Extracts:
    - Module-level docstring
    - Function definitions (name, args, docstring, line range, body preview)
    - Class definitions (name, methods, docstring, line range)
    """
    path = Path(filepath)
    if not path.exists() or path.suffix != ".py":
        return []

    try:
        file_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    try:
        tree = ast.parse(file_text, filename=filepath)
    except SyntaxError:
        return []

    chunks: List[CodeChunk] = []

    # Module-level docstring
    mod_doc = _get_docstring(tree)
    if mod_doc:
        chunks.append(
            CodeChunk(
                file_path=filepath,
                chunk_type="module",
                name=path.stem,
                signature=f"module {path.stem}",
                docstring=mod_doc,
                start_line=1,
                end_line=len(file_text.splitlines()),
                body_preview=file_text[:500],
                full_text=file_text,
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            sig = f"def {node.name}({', '.join(a.arg for a in node.args.args)}):"
            doc = _get_docstring(node)
            start = node.lineno
            end = node.end_lineno or start
            preview, full = _chunk_text(file_text, node, start, end)
            chunks.append(
                CodeChunk(
                    file_path=filepath,
                    chunk_type="function",
                    name=node.name,
                    signature=sig,
                    docstring=doc,
                    start_line=start,
                    end_line=end,
                    body_preview=preview,
                    full_text=full,
                )
            )

        elif isinstance(node, ast.AsyncFunctionDef):
            sig = f"async def {node.name}({', '.join(a.arg for a in node.args.args)}):"
            doc = _get_docstring(node)
            start = node.lineno
            end = node.end_lineno or start
            preview, full = _chunk_text(file_text, node, start, end)
            chunks.append(
                CodeChunk(
                    file_path=filepath,
                    chunk_type="function",
                    name=node.name,
                    signature=sig,
                    docstring=doc,
                    start_line=start,
                    end_line=end,
                    body_preview=preview,
                    full_text=full,
                )
            )

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(
                ast.dumps(b) if isinstance(b, ast.Name) else "" for b in node.bases
            )
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}:"
            doc = _get_docstring(node)
            start = node.lineno
            end = node.end_lineno or start
            preview, full = _chunk_text(file_text, node, start, end)
            chunks.append(
                CodeChunk(
                    file_path=filepath,
                    chunk_type="class",
                    name=node.name,
                    signature=sig,
                    docstring=doc,
                    start_line=start,
                    end_line=end,
                    body_preview=preview,
                    full_text=full,
                )
            )

    return chunks


def parse_repository(repo_path: str) -> List[CodeChunk]:
    """Walk *repo_path* and parse every Python file into CodeChunks."""
    chunks: List[CodeChunk] = []
    repo = Path(repo_path).resolve()

    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                chunks.extend(parse_file(fpath))
            except Exception:
                continue

    return chunks


def check_index_staleness(repo_path: str, index_path: str) -> bool:
    """Check if any Python file is newer than the index file.

    Returns True if index is stale or missing.
    """
    index_file = Path(index_path)
    if not index_file.exists():
        return True

    index_mtime = index_file.stat().st_mtime
    repo = Path(repo_path).resolve()

    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                if os.path.getmtime(fpath) > index_mtime:
                    return True
            except OSError:
                continue

    return False
