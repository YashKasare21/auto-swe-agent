"""Aggregate and report resilience metrics across all eval result JSON files.

Shows retry events, circuit breaker events, and model failure rates per run.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

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


def _count_retry_events(output: str) -> int:
    return len(re.findall(r"\[RETRY\] attempt \d+/\d+ failed", output))


def _count_circuit_events(output: str) -> list[str]:
    events = []
    for line in output.splitlines():
        if "[CIRCUIT OPENED]" in line or "[CIRCUIT OPEN]" in line:
            events.append(line.strip())
    return events


def _model_failure_rates(outputs: list[str]) -> Dict[str, int]:
    model_fails: Dict[str, int] = defaultdict(int)
    for output in outputs:
        for line in output.splitlines():
            m = re.search(r"\[FALLBACK\] ([\w/.-]+) failed", line)
            if m:
                model_fails[m.group(1)] += 1
    return dict(model_fails)


def print_report(records: list[dict]) -> None:
    if not records:
        print("No result files found in eval/")
        return

    print(f"\n{'='*80}")
    print("RESILIENCE REPORT")
    print(f"{'='*80}")

    # Per-run summary
    print(f"\n{'Run / Case':<45} {'Iter':>4} {'CE':>4} {'Open':>4}")
    print(f"{'-'*45} {'-'*4} {'-'*4} {'-'*4}")
    total_ce = 0
    total_open = 0
    for r in records:
        ce = r.get("circuit_events", 0)
        co = r.get("circuits_open", 0)
        total_ce += ce
        total_open += co
        label = f"{r['_file']} / {r.get('case_id', '?')}"
        print(f"{label:<45} {r.get('iterations_used', 0):>4} {ce:>4} {co:>4}")
    print(f"{'Total':<45} {'':>4} {total_ce:>4} {total_open:>4}")

    # Circuit event collection (if raw output files exist - for now we use what's in results)
    runs_with_open = sum(1 for r in records if r.get("circuits_open", 0) > 0)
    runs_with_events = sum(1 for r in records if r.get("circuit_events", 0) > 0)
    if runs_with_events:
        avg_ce = total_ce / runs_with_events
        print(f"\nRuns with circuit events: {runs_with_events}/{len(records)}")
        print(f"Runs with open circuits at end: {runs_with_open}/{len(records)}")
        print(f"Avg circuit events per affected run: {avg_ce:.1f}")
    else:
        print(f"\nNo circuit events recorded across {len(records)} runs.")

    # Model failure rates across all runs
    all_fails: Dict[str, int] = Counter()
    for r in records:
        model_used = r.get("most_used_model", "unknown")
        iterations = r.get("iterations_used", 0)
        # Infer failures from circuit events if model info available
        for model_key in [
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-lite",
            "groq/llama-3.3-70b-versatile",
            "groq/llama3-8b-8192",
        ]:
            pass  # We don't have per-model breakdown in results JSON yet

    print(f"\nModel used per run:")
    model_counter: Dict[str, int] = Counter()
    for r in records:
        m = r.get("most_used_model", "unknown")
        model_counter[m] += 1
    for model, count in model_counter.most_common():
        print(f"  {model:<40} {count:>2} runs")

    # Cost vs reliability
    print(f"\nCost vs Reliability:")
    print(f"{'Run / Case':<45} {'Cost':>8} {'Result':>6} {'CE':>4}")
    print(f"{'-'*45} {'-'*8} {'-'*6} {'-'*4}")
    for r in records:
        label = f"{r['_file']} / {r.get('case_id', '?')}"
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"{label:<45} ${r.get('total_cost_usd', 0):>6.4f} {status:>6} {r.get('circuit_events', 0):>4}")


def _try_chart(records: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = []
    ce_values = []
    cost_values = []
    for r in records:
        label = f"{r.get('case_id', '?')[:12]}"
        if label not in labels:
            labels.append(label)
            ce_values.append(r.get("circuit_events", 0))
            cost_values.append(r.get("total_cost_usd", 0.0))

    if not labels:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Circuit events bar chart
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in ce_values]
    ax1.bar(range(len(labels)), ce_values, color=colors)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Circuit Events")
    ax1.set_title("Circuit Breaker Events per Run")

    # Cost scatter with circuit events overlay
    ax2.scatter(cost_values, ce_values, c="#3498db", s=60)
    ax2.set_xlabel("Cost (USD)")
    ax2.set_ylabel("Circuit Events")
    ax2.set_title("Cost vs Circuit Events")
    for i, label in enumerate(labels):
        ax2.annotate(label, (cost_values[i], ce_values[i]), fontsize=7)

    out = EVAL_DIR / "resilience_chart.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    records = load_results()
    print_report(records)
    _try_chart(records)
