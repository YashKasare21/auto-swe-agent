"""Visualisez the LangGraph flow with the current node highlighted."""
from __future__ import annotations

from typing import Optional


def render_graph(current_node: Optional[str] = None) -> str:
    """Return an HTML+CSS flow diagram as a string.

    Nodes: planner -> executor -> verify -> git_workflow -> END.
    The *current_node* is highlighted in green; completed path is dimmed.
    """
    nodes = [
        ("planner", "Planner"),
        ("executor", "Executor"),
        ("verify", "Verify"),
        ("git_workflow", "Git Workflow"),
        ("end", "END"),
    ]

    arrows = [
        ("planner", "executor"),
        ("executor", "planner"),
        ("planner", "verify"),
        ("verify", "planner"),
        ("verify", "git_workflow"),
        ("git_workflow", "end"),
        ("planner", "end"),
    ]

    def _color(nid: str) -> str:
        if nid == current_node:
            return "#22c55e"
        return "#e5e7eb"

    def _text_color(nid: str) -> str:
        return "#ffffff" if nid == current_node else "#6b7280"

    def _glow(nid: str) -> str:
        return (
            "box-shadow: 0 0 12px rgba(34,197,94,0.6);"
            if nid == current_node
            else ""
        )

    node_divs = []
    for nid, label in nodes:
        bg = _color(nid)
        tc = _text_color(nid)
        glow = _glow(nid)
        node_divs.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'gap:4px">'
            f'<div style="background:{bg};color:{tc};padding:8px 18px;'
            f'border-radius:8px;font-weight:600;font-size:14px;'
            f'transition:all 0.3s ease;{glow}">{label}</div>'
            f"</div>"
        )

    # Simple grid layout: two rows
    # Row 1: planner --[executor loop]--> verify --[git]--> END
    # Row 2: (hidden arrows back from executor/verify to planner)
    html = f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;padding:8px 0">
      <div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap">
        {node_divs[0]}
        <div style="color:#9ca3af;font-size:20px">→</div>
        {node_divs[1]}
        <div style="color:#9ca3af;font-size:12px;writing-mode:vertical-lr;text-orientation:mixed;opacity:0.5">↻ loop</div>
        {node_divs[0]}
        <div style="color:#9ca3af;font-size:20px">→</div>
        {node_divs[2]}
        <div style="color:#9ca3af;font-size:20px">→</div>
        {node_divs[3]}
        <div style="color:#9ca3af;font-size:20px">→</div>
        {node_divs[4]}
      </div>
    </div>
    """
    return html
