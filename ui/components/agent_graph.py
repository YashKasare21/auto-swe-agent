"""Visualise the LangGraph flow with the current node highlighted."""

from __future__ import annotations

from typing import Optional


def render_graph(
    current_node: Optional[str] = None, state: Optional[dict] = None
) -> str:
    """Return an HTML+CSS flow diagram as a string.

    Supports both single-agent (planner → executor ↔ verify → git)
    and multi-agent (manager → planner → coder ↔ executor → verify → reviewer → git) flows.

    Determines mode from the state's `current_agent` field, or by checking if
    the current_node is specific to multi-agent mode.
    """
    if state and state.get("current_agent") in ("manager", "coder", "reviewer"):
        is_multi = True
    elif current_node in ("manager", "coder", "reviewer"):
        is_multi = True
    else:
        is_multi = False

    if is_multi:
        nodes = [
            ("manager", "Manager"),
            ("planner", "Planner"),
            ("coder", "Coder"),
            ("executor", "Executor"),
            ("verify", "Verify"),
            ("reviewer", "Reviewer"),
            ("git_workflow", "Git Workflow"),
            ("end", "END"),
        ]
    else:
        nodes = [
            ("planner", "Planner"),
            ("executor", "Executor"),
            ("verify", "Verify"),
            ("git_workflow", "Git Workflow"),
            ("end", "END"),
        ]

    def _color(nid: str) -> str:
        if nid == current_node:
            return "#22c55e"
        if nid == "end":
            return "#f3f4f6"
        return "#e5e7eb"

    def _text_color(nid: str) -> str:
        return "#ffffff" if nid == current_node else "#6b7280"

    def _glow(nid: str) -> str:
        return (
            "box-shadow: 0 0 12px rgba(34,197,94,0.6);" if nid == current_node else ""
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
            f"border-radius:8px;font-weight:600;font-size:14px;"
            f'transition:all 0.3s ease;{glow}">{label}</div>'
            f"</div>"
        )

    # Build arrows between consecutive nodes
    arrow_divs = []
    for i in range(len(node_divs) - 1):
        arrow_divs.append(node_divs[i])
        # Add loop indicator between coder↔executor
        if is_multi and nodes[i][0] == "executor":
            arrow_divs.append(
                '<div style="color:#9ca3af;font-size:12px;writing-mode:vertical-lr;'
                'text-orientation:mixed;opacity:0.5">↻ loop</div>'
            )
            arrow_divs.append(node_divs[i])
        arrow_divs.append('<div style="color:#9ca3af;font-size:20px">→</div>')
    arrow_divs.append(node_divs[-1])

    html = f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;padding:8px 0">
      <div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap">
        {''.join(arrow_divs)}
      </div>
    </div>
    """
    return html
