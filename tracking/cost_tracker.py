"""Cost and token tracking for auto-swe-agent LLM calls."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class CostConfig:
    model_name: str
    input_cost_per_1k: float  # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens


# Approximate pricing (USD). Update as providers change rates.
MODEL_COSTS: dict[str, CostConfig] = {
    "gemini/gemini-2.0-flash": CostConfig("gemini/gemini-2.0-flash", 0.000075, 0.0003),
    "gemini/gemini-2.0-flash-lite": CostConfig(
        "gemini/gemini-2.0-flash-lite", 0.0000375, 0.00015
    ),
    "groq/llama-3.3-70b-versatile": CostConfig(
        "groq/llama-3.3-70b-versatile", 0.00059, 0.00079
    ),
    "groq/llama3-8b-8192": CostConfig("groq/llama3-8b-8192", 0.00005, 0.0001),
}


@dataclass
class CostRecord:
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str
    node_type: str  # e.g. "planner"
    estimated: bool  # True if token counts were estimated, not from API


class CostTracker:
    def __init__(self, budget_usd: float = 5.0):
        self.budget_usd = budget_usd
        self.records: list[CostRecord] = []
        self.total_cost: float = 0.0

    def add_call(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        node_type: str = "planner",
        estimated: bool = False,
    ) -> float:
        """Record an LLM call. Returns the cost of this call in USD."""
        config = MODEL_COSTS.get(model_name)
        if config:
            cost = (input_tokens / 1000) * config.input_cost_per_1k + (
                output_tokens / 1000
            ) * config.output_cost_per_1k
        else:
            cost = 0.0  # unknown model — don't guess

        record = CostRecord(
            model_used=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 8),
            timestamp=datetime.utcnow().isoformat(),
            node_type=node_type,
            estimated=estimated,
        )
        self.records.append(record)
        self.total_cost += cost
        return cost

    def get_total_cost(self) -> float:
        return round(self.total_cost, 6)

    def get_total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def get_model_breakdown(self) -> dict[str, dict]:
        breakdown: dict[str, dict] = {}
        for r in self.records:
            m = breakdown.setdefault(
                r.model_used, {"calls": 0, "tokens": 0, "cost": 0.0}
            )
            m["calls"] += 1
            m["tokens"] += r.input_tokens + r.output_tokens
            m["cost"] = round(m["cost"] + r.cost_usd, 8)
        return breakdown

    def get_summary(self) -> dict:
        breakdown = self.get_model_breakdown()
        most_used = (
            max(breakdown, key=lambda m: breakdown[m]["calls"]) if breakdown else None
        )
        return {
            "total_cost_usd": self.get_total_cost(),
            "total_calls": len(self.records),
            "total_tokens": self.get_total_tokens(),
            "model_breakdown": breakdown,
            "most_used_model": most_used,
            "budget_usd": self.budget_usd,
            "budget_exceeded": self.check_budget_exceeded(),
        }

    def check_budget_exceeded(self) -> bool:
        """Returns True if total cost has exceeded the budget (ignored if budget=0)."""
        return self.budget_usd > 0 and self.total_cost > self.budget_usd

    def export_json(self, filepath: str | Path) -> None:
        Path(filepath).write_text(
            json.dumps(
                {
                    "summary": self.get_summary(),
                    "records": [asdict(r) for r in self.records],
                },
                indent=2,
            )
        )

    def reset(self) -> None:
        self.records.clear()
        self.total_cost = 0.0
