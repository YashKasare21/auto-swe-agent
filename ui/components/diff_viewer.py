"""Diff viewer component for showing before/after file changes."""
from __future__ import annotations

import difflib


def render_diff(old_text: str, new_text: str, filepath: str = "") -> str:
    """Return a syntax-highlighted unified diff as HTML."""
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    lines = list(diff)

    if not lines:
        return "<p style='color:#6b7280;font-style:italic'>No changes.</p>"

    html_parts = [
        "<div style='font-family:monospace;font-size:13px;line-height:1.6;"
        "background:#1e1e2e;color:#cdd6f4;padding:12px;border-radius:8px;"
        "overflow-x:auto;white-space:pre'>"
    ]
    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            cls = "color:#6c7086;font-weight:600"
        elif line.startswith("@@"):
            cls = "color:#f5c2e7;font-weight:600"
        elif line.startswith("+"):
            cls = "color:#a6e3a1"
        elif line.startswith("-"):
            cls = "color:#f38ba8"
        else:
            cls = "color:#cdd6f4"
        html_parts.append(f"<div style='{cls}'>{line.rstrip()}</div>")
    html_parts.append("</div>")
    return "".join(html_parts)
