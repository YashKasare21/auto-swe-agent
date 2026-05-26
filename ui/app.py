"""Streamlit UI dashboard for auto-swe-agent.

Pages:
  - Run Agent   : execute a new task via subprocess
  - Live Monitor : real-time execution view (polls state file)
  - Results      : historical eval results + charts
  - Costs        : cost analysis dashboard
  - System Status: circuit breaker + model health
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.components.agent_graph import render_graph
from ui.components.cost_chart import (
    budget_gauge,
    cost_bar_chart,
    cost_pie_chart,
    model_usage_stacked_bar,
)
from ui.state_manager import AgentStateManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

st.set_page_config(
    page_title="auto-swe-agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "agent_process" not in st.session_state:
    st.session_state.agent_process = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "agent_start_time" not in st.session_state:
    st.session_state.agent_start_time = None
if "agent_complete" not in st.session_state:
    st.session_state.agent_complete = False

_state_mgr = AgentStateManager()

FALLBACK_MODELS = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "groq/llama-3.3-70b-versatile",
    "groq/llama3-8b-8192",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_eval_results() -> list[dict]:
    records: list[dict] = []
    for path in sorted(EVAL_DIR.glob("results_*.json")):
        try:
            data = json.loads(path.read_text())
            for r in data:
                r["_file"] = path.name
                r["_timestamp"] = datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat()
            records.extend(data)
        except (json.JSONDecodeError, OSError):
            pass
    return records


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _model_icon(state: str) -> str:
    return {"closed": "🟢", "half-open": "🟡", "open": "🔴"}.get(state, "⚪")


def _status_box(
    label: str,
    value: str,
    color: str = "#6b7280",
) -> str:
    return (
        f'<div style="background:{color}15;border:1px solid {color}40;'
        f'border-radius:8px;padding:12px 16px;text-align:center">'
        f'<div style="font-size:12px;color:{color}80;text-transform:uppercase;'
        f'letter-spacing:0.5px">{label}</div>'
        f'<div style="font-size:22px;font-weight:700;color:{color};'
        f'margin-top:4px">{value}</div></div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/fluency/96/robot.png",
    width=48,
)
st.sidebar.title("auto-swe-agent")
st.sidebar.caption("Autonomous code-fixing agent")

page = st.sidebar.radio(
    "Navigation",
    ["🚀 Run Agent", "📊 Live Monitor", "📈 Results", "💰 Costs", "🔧 System Status"],
    label_visibility="collapsed",
)

st.sidebar.divider()
_state = _state_mgr.load_state()
if _state and _state_mgr.is_running():
    st.sidebar.success(f"Agent running (iter {_state.get('iteration_count', 0)})")
    st.sidebar.progress(min(_state.get("iteration_count", 0) / 15, 1.0))
elif _state and _state.get("status") == "completed":
    st.sidebar.info("Last run completed")
    if _state.get("tests_passed") is True:
        st.sidebar.success("Tests passed")
    elif _state.get("tests_passed") is False:
        st.sidebar.error("Tests failed")
else:
    st.sidebar.info("No agent running")

# ---------------------------------------------------------------------------
# PAGE: Run Agent
# ---------------------------------------------------------------------------

if page == "🚀 Run Agent":
    st.title("🚀 Run Agent")
    st.markdown("Describe the issue you want the agent to fix.")

    col1, col2 = st.columns([3, 1])
    with col1:
        issue = st.text_area(
            "Issue description",
            placeholder="e.g. Fix the authentication bug in login.py — passwords are not being hashed before storage.",
            height=160,
            label_visibility="collapsed",
        )
    with col2:
        model_choice = st.selectbox(
            "Primary model",
            FALLBACK_MODELS,
            index=0,
        )
        budget = st.slider("Budget ($)", 0.0, 20.0, 5.0, 0.5)
        workspace = st.text_input("Workspace", value=str(PROJECT_ROOT))
        single_agent = st.checkbox(
            "Single-agent mode",
            value=False,
            help="Use legacy single-agent (planner-only) instead of multi-agent pipeline",
        )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        retry_max = st.number_input("Retries", min_value=0, max_value=10, value=3)
    with col_b:
        retry_delay = st.number_input(
            "Retry delay (s)", min_value=0.5, max_value=30.0, value=2.0, step=0.5
        )
    with col_c:
        circuit_threshold = st.number_input(
            "Circuit threshold", min_value=1, max_value=20, value=5
        )
    with col_d:
        circuit_timeout = st.number_input(
            "Circuit timeout (s)", min_value=30, max_value=600, value=300, step=30
        )

    # Disable fallback models by setting budget=0 for models we don't want
    fallback_order = FALLBACK_MODELS[:]
    idx = fallback_order.index(model_choice)
    st.info(f"Fallback chain: **{' → '.join(fallback_order[idx:])}**")

    run_disabled = not issue.strip() or (
        st.session_state.agent_process is not None
        and st.session_state.agent_process.poll() is None
    )

    if st.button(
        "▶ Run Agent", type="primary", disabled=run_disabled, use_container_width=True
    ):
        _state_mgr.clear()
        st.session_state.agent_logs = []
        st.session_state.agent_complete = False
        st.session_state.agent_start_time = time.time()

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "agent.py"),
            issue.strip(),
            "--workspace",
            workspace,
            "--budget",
            str(budget),
            "--retry-max",
            str(retry_max),
            "--retry-delay",
            str(retry_delay),
            "--circuit-threshold",
            str(circuit_threshold),
            "--circuit-timeout",
            str(circuit_timeout),
        ]
        if single_agent:
            cmd.append("--single-agent")
        st.session_state.agent_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        st.rerun()

    # Live output area
    log_area = st.empty()
    status_area = st.empty()
    cost_area = st.empty()

    proc = st.session_state.agent_process
    if proc is not None:
        if proc.poll() is None:
            # Still running — stream output
            try:
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        st.session_state.agent_logs.append(line.rstrip())
                        # Keep last 200 lines
                        if len(st.session_state.agent_logs) > 200:
                            st.session_state.agent_logs = st.session_state.agent_logs[
                                -200:
                            ]
            except (ValueError, OSError):
                pass

        elapsed = (
            time.time() - st.session_state.agent_start_time
            if st.session_state.agent_start_time
            else 0
        )
        status_area.info(
            f"Running for {_format_time(elapsed)} — "
            f"{len(st.session_state.agent_logs)} log lines captured"
        )

        recent = st.session_state.agent_logs[-50:]
        log_area.code(
            "\n".join(recent) if recent else "Waiting for output...",
            language="",
        )

        # Check if process finished
        retcode = proc.poll()
        if retcode is not None:
            st.session_state.agent_complete = True
            st.session_state.agent_process = None

            # Read final state
            final_state = _state_mgr.load_state()
            st.success(f"Agent exited with code {retcode}")

            if final_state:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tests Passed", final_state.get("tests_passed"))
                with col2:
                    st.metric("Iterations", final_state.get("iteration_count", 0))
                with col3:
                    st.metric(
                        "Total Cost", f"${final_state.get('total_cost_usd', 0):.4f}"
                    )
                with col4:
                    bn = final_state.get("branch_name") or "—"
                    st.metric("Branch", bn)

                circuit = final_state.get("circuit_status", {})
                if circuit:
                    st.subheader("Circuit Breaker Status")
                    for model, info in circuit.items():
                        st.write(
                            f"{_model_icon(info['state'])} **{model}** — {info['state']} ({info['failures']} failures)"
                        )

    if st.session_state.agent_complete:
        if st.button("🔄 Clear & New Run", use_container_width=True):
            st.session_state.agent_logs = []
            st.session_state.agent_complete = False
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Live Monitor
# ---------------------------------------------------------------------------

elif page == "📊 Live Monitor":
    st.title("📊 Live Monitor")
    st.caption("Auto-refreshes every 2 seconds while the agent is running.")

    placeholder = st.empty()

    state = _state_mgr.load_state()
    is_running = _state_mgr.is_running()

    if not state:
        placeholder.info(
            "No agent state found. Start a run from the *Run Agent* page "
            "to see live data here."
        )
    else:
        with placeholder.container():
            # Node diagram
            st.subheader("Current Node")
            current_node = state.get("current_node", "idle")
            st.markdown(render_graph(current_node, state), unsafe_allow_html=True)

            # Current agent badge
            agent_colors = {
                "manager": "#3b82f6",
                "planner": "#8b5cf6",
                "coder": "#f59e0b",
                "reviewer": "#ef4444",
                "executor": "#10b981",
                "verify": "#06b6d4",
                "git_workflow": "#6b7280",
                None: "#6b7280",
                "idle": "#6b7280",
            }
            current_agent = state.get("current_agent") or "idle"
            agent_color = agent_colors.get(current_agent, "#6b7280")
            agent_label = (
                current_agent.upper().replace("_", " ")
                if current_agent != "idle"
                else "IDLE"
            )
            st.markdown(
                f'<div style="display:inline-block;background:{agent_color}20;'
                f"border:2px solid {agent_color};border-radius:20px;padding:6px 18px;"
                f'font-weight:700;font-size:16px;color:{agent_color}">{agent_label}</div>',
                unsafe_allow_html=True,
            )

            # Key metrics
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Iteration", state.get("iteration_count", 0))
            with k2:
                tp = state.get("tests_passed")
                tp_icon = {
                    True: "🟢 Passed",
                    False: "🔴 Failed",
                    None: "🟡 Pending",
                }.get(tp, "—")
                st.metric("Tests", tp_icon)
            with k3:
                st.metric(
                    "Verification Attempts", state.get("verification_attempts", 0)
                )
            with k4:
                st.metric("Total Cost", f"${state.get('total_cost_usd', 0):.4f}")
            with k5:
                st.metric("Messages", state.get("messages_count", 0))

            # Model info
            col_l, col_r = st.columns(2)
            with col_l:
                last_model = state.get("last_model_used", "unknown")
                st.info(f"**Last model used:** {last_model}")

                model_breakdown = state.get("model_breakdown", {})
                if model_breakdown:
                    st.subheader("Cost by Model")
                    fig = cost_pie_chart(model_breakdown, title="")
                    st.plotly_chart(fig, use_container_width=True)

            with col_r:
                budget = state.get("budget_usd", 5.0)
                cost = state.get("total_cost_usd", 0.0)
                if budget > 0:
                    fig = budget_gauge(cost, budget)
                    st.plotly_chart(fig, use_container_width=True)

            # Circuit breaker status
            circuit = state.get("circuit_status", {})
            if circuit:
                st.subheader("Circuit Breaker Status")
                ccols = st.columns(len(circuit))
                for i, (model, info) in enumerate(circuit.items()):
                    icon = _model_icon(info["state"])
                    with ccols[i]:
                        st.markdown(
                            f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
                            f"padding:12px;text-align:center'>"
                            f"<div style='font-size:24px'>{icon}</div>"
                            f"<div style='font-weight:600;font-size:13px;margin-top:4px'>{model.split('/')[-1]}</div>"
                            f"<div style='font-size:11px;color:#6b7280'>{info['state']} · {info['failures']} failures</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Recent circuit events
            events = state.get("circuit_events", [])
            if events:
                with st.expander(f"Circuit Events ({len(events)})", expanded=False):
                    for ev in events[-10:]:
                        st.code(ev)

        if is_running:
            time.sleep(2)
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Results
# ---------------------------------------------------------------------------

elif page == "📈 Results":
    st.title("📈 Results")
    st.caption("Historical evaluation results")

    results = _load_eval_results()

    if not results:
        st.info("No eval results found. Run `python eval/run_eval.py` first.")
    else:
        df = pd.DataFrame(results)

        _MODEL_COL = "most_used_model"
        _COST_COL = "total_cost_usd"
        _TOKEN_COL = "total_tokens"
        _CASE_COL = "case_id"
        _ITER_COL = "iterations_used"
        _PASSED_COL = "passed"
        _TS_COL = "_timestamp"
        _TS_DT_COL = "_ts_dt"

        expected_cols = [
            _TS_COL,
            _MODEL_COL,
            _COST_COL,
            _CASE_COL,
            _ITER_COL,
            _PASSED_COL,
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing or df.empty:
            st.warning(
                f"Eval result data is incomplete or empty. "
                f"Missing columns: {', '.join(missing) if missing else 'none'}. "
                "Displaying available data."
            )
            for col in [_TS_COL, _MODEL_COL]:
                if col not in df.columns:
                    df[col] = ""
            for col in [_COST_COL, _TOKEN_COL, _ITER_COL]:
                if col not in df.columns:
                    df[col] = 0.0
            if _PASSED_COL not in df.columns:
                df[_PASSED_COL] = False
            if _CASE_COL not in df.columns:
                df[_CASE_COL] = "unknown"
            if _TS_DT_COL not in df.columns:
                df[_TS_DT_COL] = pd.Timestamp.now()

        if _TS_COL in df.columns:
            df[_TS_DT_COL] = pd.to_datetime(df[_TS_COL], errors="coerce")

        model_options = (
            sorted(df[_MODEL_COL].dropna().unique()) if _MODEL_COL in df.columns else []
        )

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            model_filter = st.multiselect("Model", options=model_options)
        with col_f2:
            status_filter = st.multiselect("Status", options=["PASS", "FAIL"])
        with col_f3:
            if not model_filter and model_options:
                model_filter = model_options
            else:
                model_filter = model_filter or model_options
            status_filter = status_filter or ["PASS", "FAIL"]
        with col_f4:
            sort_by = st.selectbox(
                "Sort by",
                ["timestamp", "cost", "iterations", "tokens"],
                index=0,
            )

        filtered = df.copy()
        if _MODEL_COL in filtered.columns and model_filter:
            filtered = filtered[filtered[_MODEL_COL].isin(model_filter)]
        if _PASSED_COL in filtered.columns:
            boolean_status = [s == "PASS" for s in status_filter]
            filtered = filtered[filtered[_PASSED_COL].isin(boolean_status)]

        sort_map = {
            "timestamp": _TS_DT_COL,
            "cost": _COST_COL,
            "iterations": _ITER_COL,
            "tokens": _TOKEN_COL,
        }
        sort_key = sort_map.get(sort_by, _TS_DT_COL)
        if sort_key in filtered.columns:
            filtered = filtered.sort_values(sort_key, ascending=False)

        display_cols = [
            _CASE_COL,
            _PASSED_COL,
            _ITER_COL,
            "model_used",
            _COST_COL,
            _TOKEN_COL,
            "verification_attempts",
            "circuit_events",
            "branch_name",
            "_file",
        ]
        available = [c for c in display_cols if c in filtered.columns]
        if not filtered.empty and available:
            st.dataframe(
                filtered[available],
                use_container_width=True,
                column_config={
                    _PASSED_COL: st.column_config.CheckboxColumn("Passed"),
                    _COST_COL: st.column_config.NumberColumn("Cost", format="$%.4f"),
                    _TOKEN_COL: st.column_config.NumberColumn("Tokens", format="%d"),
                    "circuit_events": st.column_config.NumberColumn("CE"),
                    "branch_name": st.column_config.TextColumn("Branch"),
                },
                hide_index=True,
            )
        else:
            st.info("No matching results to display.")

        st.subheader("Charts")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Success Rate", "Cost per Run", "Model Usage", "Iterations vs Success"]
        )

        with tab1:
            if len(filtered) > 1 and _PASSED_COL in filtered.columns:
                success_rate = (
                    filtered[_PASSED_COL].rolling(5, min_periods=1).mean() * 100
                )
                fig = px.line(
                    y=success_rate,
                    title="Success Rate (rolling avg, last 5 runs)",
                    labels={"index": "Run", "y": "Success Rate (%)"},
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 data points for chart.")

        with tab2:
            if (
                not filtered.empty
                and _COST_COL in filtered.columns
                and _CASE_COL in filtered.columns
            ):
                fig = cost_bar_chart(
                    filtered[_COST_COL].tolist(),
                    filtered[_CASE_COL].tolist(),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data.")

        with tab3:
            if not filtered.empty and _MODEL_COL in filtered.columns:
                usage = filtered[_MODEL_COL].value_counts()
                fig = px.pie(
                    values=usage.values,
                    names=usage.index,
                    title="Model Usage Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data.")

        with tab4:
            if (
                len(filtered) > 1
                and _ITER_COL in filtered.columns
                and _COST_COL in filtered.columns
            ):
                fig = px.scatter(
                    filtered,
                    x=_ITER_COL,
                    y=_COST_COL,
                    color=_PASSED_COL if _PASSED_COL in filtered.columns else None,
                    title="Iterations vs Cost",
                    labels={
                        _ITER_COL: "Iterations",
                        _COST_COL: "Cost (USD)",
                    },
                    hover_data=[_CASE_COL] if _CASE_COL in filtered.columns else None,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 data points.")

# ---------------------------------------------------------------------------
# PAGE: Costs
# ---------------------------------------------------------------------------

elif page == "💰 Costs":
    st.title("💰 Costs")
    results = _load_eval_results()

    if not results:
        st.info("No cost data available yet.")
    else:
        total_cost = sum(r.get("total_cost_usd", 0) for r in results)
        total_tokens = sum(r.get("total_tokens", 0) for r in results)

        passed_runs = [r for r in results if r.get("passed")]
        failed_runs = [r for r in results if not r.get("passed")]
        avg_passed = (
            sum(r.get("total_cost_usd", 0) for r in passed_runs) / len(passed_runs)
            if passed_runs
            else 0
        )
        avg_failed = (
            sum(r.get("total_cost_usd", 0) for r in failed_runs) / len(failed_runs)
            if failed_runs
            else 0
        )

        most_expensive = max(results, key=lambda r: r.get("total_cost_usd", 0))
        me_case = most_expensive.get("case_id", "?")
        me_cost = most_expensive.get("total_cost_usd", 0)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Spent", f"${total_cost:.4f}")
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("Avg (Passed)", f"${avg_passed:.4f}")
        with col4:
            st.metric("Avg (Failed)", f"${avg_failed:.4f}")
        with col5:
            st.metric("Most Expensive", f"${me_cost:.4f}", me_case)

        last_budget = results[-1].get("total_cost_usd", 5.0)
        if last_budget > 0:
            fig = budget_gauge(total_cost, max(total_cost * 1.2, last_budget))
            st.plotly_chart(fig, use_container_width=True)

        tab_c1, tab_c2, tab_c3 = st.tabs(
            ["Run Costs", "By Model", "Cost vs Iterations"]
        )
        with tab_c1:
            df = pd.DataFrame(results)
            cost_col = "total_cost_usd"
            case_col = "case_id"
            fig = cost_bar_chart(
                df[cost_col].tolist() if cost_col in df.columns else [],
                df[case_col].tolist() if case_col in df.columns else [],
                "Cost per Run",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_c2:
            model_breakdown: Dict[str, dict] = {}
            for r in results:
                m = r.get("most_used_model", "unknown")
                entry = model_breakdown.setdefault(
                    m, {"calls": 0, "tokens": 0, "cost": 0.0}
                )
                entry["calls"] += 1
                entry["tokens"] += r.get("total_tokens", 0)
                entry["cost"] += r.get("total_cost_usd", 0)

            fig = cost_pie_chart(model_breakdown, "Cost by Model")
            st.plotly_chart(fig, use_container_width=True)

        with tab_c3:
            if len(results) > 1:
                df = pd.DataFrame(results)
                x_col = "total_cost_usd"
                y_col = "iterations_used"
                c_col = "passed"
                if x_col in df.columns and y_col in df.columns:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=c_col if c_col in df.columns else None,
                        hover_data=["case_id"] if "case_id" in df.columns else None,
                        labels={x_col: "Cost (USD)", y_col: "Iterations"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if st.button("📥 Download Cost Data as CSV", use_container_width=True):
            df = pd.DataFrame(results)
            export_cols = [
                "case_id",
                "passed",
                "total_cost_usd",
                "total_tokens",
                "iterations_used",
                "most_used_model",
            ]
            available = [c for c in export_cols if c in df.columns]
            csv = df[available].to_csv(index=False)
            st.download_button(
                "Confirm Download",
                data=csv,
                file_name=f"cost_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# PAGE: System Status
# ---------------------------------------------------------------------------

elif page == "🔧 System Status":
    st.title("🔧 System Status")
    st.caption("Circuit breaker, model health, and Docker sandbox status")

    # Load from latest agent state
    state = _state_mgr.load_state()

    # ---- Circuit Breaker ----
    st.subheader("⚡ Circuit Breaker")

    circuit = (state or {}).get("circuit_status", {})
    if circuit:
        cb_df = pd.DataFrame(
            [
                {
                    "Model": m,
                    "State": info["state"],
                    "Failures": info["failures"],
                }
                for m, info in circuit.items()
            ]
        )
        st.dataframe(cb_df, use_container_width=True, hide_index=True)

        cols = st.columns(len(circuit))
        for i, (model, info) in enumerate(circuit.items()):
            with cols[i]:
                icon = _model_icon(info["state"])
                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb;border-radius:8px;
                               padding:16px;text-align:center">
                        <div style="font-size:32px">{icon}</div>
                        <div style="font-weight:600;margin:4px 0">{model.split('/')[-1]}</div>
                        <div style="font-size:12px;color:{'#ef4444' if info['state'] == 'open' else '#6b7280'}">
                            {info['state'].upper()}
                        </div>
                        <div style="font-size:11px;color:#9ca3af">
                            {info['failures']} consecutive failure{'s' if info['failures'] != 1 else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No circuit breaker data available yet (no agent runs recorded).")

    # ---- Model Health ----
    st.subheader("📊 Model Health")
    results = _load_eval_results()

    if results:
        model_runs: Dict[str, list] = {}
        for r in results:
            m = r.get("most_used_model", "unknown")
            model_runs.setdefault(m, []).append(r.get("passed", False))

        health_data = []
        for model, outcomes in model_runs.items():
            recent = outcomes[-10:]
            success_rate = sum(recent) / len(recent) * 100 if recent else 0
            health_data.append(
                {
                    "Model": model,
                    "Runs": len(recent),
                    "Success Rate": f"{success_rate:.0f}%",
                    "Bar": (
                        "🟢"
                        if success_rate >= 80
                        else ("🟡" if success_rate >= 50 else "🔴")
                    ),
                }
            )

        if health_data:
            st.dataframe(
                pd.DataFrame(health_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Bar": st.column_config.TextColumn("Health", width="small")
                },
            )
    else:
        st.info("No model health data available.")

    # ---- Retry Statistics ----
    st.subheader("🔄 Retry Statistics")
    if results:
        total_ce = sum(r.get("circuit_events", 0) for r in results)
        total_runs = len(results)
        st.metric("Total Circuit Events", total_ce)
        st.metric(
            "Runs with Circuit Events",
            sum(1 for r in results if r.get("circuit_events", 0) > 0),
        )
        st.metric(
            "Runs with Open Circuits",
            sum(1 for r in results if r.get("circuits_open", 0) > 0),
        )
    else:
        st.info("No retry data available.")

    # ---- Docker Sandbox ----
    st.subheader("🐳 Docker Sandbox")

    def _check_docker_sandbox() -> None:
        try:
            import docker

            client = docker.from_env()
        except (docker.errors.DockerException, FileNotFoundError, ImportError) as e:
            st.markdown(
                f'<div style="border:1px solid #f59e0b40;border-radius:8px;'
                f'background:#fef3c715;padding:16px">'
                f'<div style="font-size:14px;font-weight:600;color:#d97706">'
                f"🐳 Runtime: Container Sandbox Interface Mode</div>"
                f'<div style="font-size:13px;color:#92400e;margin-top:4px">'
                f"Fallback to isolated local shell context due to environment "
                f"security constraints. Docker socket unreachable: {e}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            return

        try:
            containers = client.containers.list(
                filters={"label": "role=auto-swe-agent-sandbox"},
                all=True,
            )
        except Exception as e:
            st.markdown(
                f'<div style="border:1px solid #f59e0b40;border-radius:8px;'
                f'background:#fef3c715;padding:16px">'
                f'<div style="font-size:14px;font-weight:600;color:#d97706">'
                f"🐳 Runtime: Container Sandbox Interface Mode</div>"
                f'<div style="font-size:13px;color:#92400e;margin-top:4px">'
                f"Fallback to isolated local shell context due to environment "
                f"security constraints. Sandbox query failed: {e}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            return

        if containers:
            for c in containers:
                status_icon = "🟢" if c.status == "running" else "🔴"
                st.markdown(
                    f"{status_icon} **{c.short_id}** — {c.status} "
                    f"(created {c.attrs.get('Created', '?')[:19]})"
                )
        else:
            st.info("No sandbox containers found. Start an agent run to create one.")

    _check_docker_sandbox()
