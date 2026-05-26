"""Reusable cost visualisation components using Plotly."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def cost_bar_chart(
    costs: List[float],
    labels: List[str],
    title: str = "Cost per Run",
) -> go.Figure:
    df = pd.DataFrame({"label": labels, "cost_usd": costs})
    fig = px.bar(
        df,
        x="label",
        y="cost_usd",
        title=title,
        color="cost_usd",
        color_continuous_scale="Blues",
        labels={"cost_usd": "Cost (USD)", "label": ""},
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        height=350,
        margin=dict(l=40, r=20, t=40, b=60),
    )
    return fig


def cost_pie_chart(
    breakdown: Dict[str, dict],
    title: str = "Cost by Model",
) -> go.Figure:
    if not breakdown:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig
    models = list(breakdown.keys())
    costs = [breakdown[m]["cost"] for m in models]
    calls = [breakdown[m]["calls"] for m in models]
    df = pd.DataFrame(
        {
            "model": models,
            "cost_usd": costs,
            "calls": calls,
        }
    )
    fig = px.pie(
        df,
        names="model",
        values="cost_usd",
        title=title,
        hover_data={"calls": True},
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def budget_gauge(
    current_cost: float,
    budget: float,
    title: str = "Budget Utilisation",
) -> go.Figure:
    pct = min(current_cost / budget * 100, 100) if budget > 0 else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_cost,
            number={"suffix": " USD", "font": {"size": 24}},
            delta={"reference": budget, "valueformat": ".2f"},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, budget], "tickprefix": "$"},
                "bar": {
                    "color": (
                        "#22c55e" if pct < 80 else "#eab308" if pct < 100 else "#ef4444"
                    )
                },
                "steps": [
                    {"range": [0, budget * 0.8], "color": "#f0fdf4"},
                    {"range": [budget * 0.8, budget], "color": "#fef9c3"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": budget,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=30, b=30))
    return fig


def model_usage_stacked_bar(
    records_list: List[Dict],
    title: str = "Cost by Model per Run",
) -> go.Figure:
    if not records_list:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig
    df = pd.DataFrame(records_list)
    model_cols = [c for c in df.columns if c.startswith("model_cost_")]
    if not model_cols:
        return cost_bar_chart(
            df.get("total_cost_usd", [0]),
            df.get("case_id", ["run"]),
            title,
        )
    fig = px.bar(
        df,
        x="case_id",
        y=model_cols,
        title=title,
        labels={"value": "Cost (USD)", "case_id": "", "variable": "Model"},
    )
    fig.update_layout(
        barmode="stack",
        xaxis_tickangle=-30,
        height=350,
        margin=dict(l=40, r=20, t=40, b=60),
    )
    return fig
