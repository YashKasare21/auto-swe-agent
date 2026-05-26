"""Aggregate and report costs across all eval result JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent / "eval"


def load_results() -> list[dict]:
    records = []
    for path in sorted(EVAL_DIR.glob("results_*.json")):
        try:
            data = json.loads(path.read_text())
            for r in data:
                r["_file"] = path.name
            records.extend(data)
        except (json.JSONDecodeError, KeyError):
            print(f"[WARN] Could not parse {path.name}")
    return records


def print_report(records: list[dict]) -> None:
    if not records:
        print("No result files found in eval/")
        return

    total_cost = sum(r.get("total_cost_usd", 0.0) for r in records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    costs = [
        (r["_file"] + " / " + r.get("case_id", "?"), r.get("total_cost_usd", 0.0))
        for r in records
    ]
    costs_sorted = sorted(costs, key=lambda x: x[1], reverse=True)

    avg = total_cost / len(records) if records else 0.0

    print(f"\n{'='*60}")
    print("COST REPORT")
    print(f"{'='*60}")
    print(f"{'Run / Case':<45} {'Cost':>8}")
    print(f"{'-'*45} {'-'*8}")
    for label, cost in costs_sorted:
        flag = " !" if cost > 5.0 else ""
        print(f"{label:<45} ${cost:>6.4f}{flag}")
    print(f"\nTotal runs : {len(records)}")
    print(f"Total cost : ${total_cost:.4f}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg / run  : ${avg:.4f}")
    if costs_sorted:
        print(f"Most expensive: {costs_sorted[0][0]}  (${costs_sorted[0][1]:.4f})")
        print(f"Cheapest      : {costs_sorted[-1][0]}  (${costs_sorted[-1][1]:.4f})")

    _try_chart(records)


def _try_chart(records: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[INFO] matplotlib not installed — skipping chart. Run: pip install matplotlib"
        )
        return

    labels = [r.get("case_id", "?")[:20] + f"\n({r['_file'][-15:]})" for r in records]
    costs = [r.get("total_cost_usd", 0.0) for r in records]

    fig, ax = plt.subplots(figsize=(max(6, len(records) * 1.2), 4))
    bars = ax.bar(
        range(len(labels)),
        costs,
        color=["#e74c3c" if c > 5.0 else "#3498db" for c in costs],
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Cost (USD)")
    ax.set_title("auto-swe-agent: Cost per Eval Run")
    ax.axhline(5.0, color="red", linestyle="--", linewidth=0.8, label="$5 budget")
    ax.legend()
    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0002,
            f"${cost:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    out = EVAL_DIR / "cost_chart.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    records = load_results()
    print_report(records)
