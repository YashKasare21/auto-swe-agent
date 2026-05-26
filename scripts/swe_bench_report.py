#!/usr/bin/env python3
"""Generate a markdown report from SWE-bench evaluation result files.

Usage:
    python scripts/swe_bench_report.py
    python scripts/swe_bench_report.py --results swe_bench/results/swe_bench_results_*.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(
    pattern: str = "swe_bench/results/swe_bench_results_*.json",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in sorted(Path(".").glob(pattern)):
        data = json.loads(path.read_text())
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
    return results


def generate_report(results: Optional[List[Dict[str, Any]]] = None) -> str:
    if results is None:
        results = load_results()

    if not results:
        return "No SWE-bench results found."

    lines: List[str] = []

    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    errors = sum(1 for r in results if "error" in r)
    timed_out = sum(1 for r in results if r.get("agent_result", {}).get("timed_out"))
    rate = successful / total * 100 if total > 0 else 0.0

    total_cost = 0.0
    total_tokens = 0
    for r in results:
        agent = r.get("agent_result", {})
        if isinstance(agent, dict):
            try:
                total_cost += float(agent.get("summary", {}).get("total_cost_usd", 0))
            except (ValueError, TypeError):
                pass
            try:
                total_tokens += int(agent.get("summary", {}).get("total_tokens", 0))
            except (ValueError, TypeError):
                pass

    lines.append("# SWE-bench Lite Evaluation Report")
    lines.append("")
    lines.append(
        f"**Date:** {Path('swe_bench/results').glob('*.json') and 'see file timestamps' or 'N/A'}"
    )
    lines.append(f"**Agent:** auto-swe-agent (multi-agent architecture)")
    lines.append(
        f"**Dataset:** [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)"
    )
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tasks | {total} |")
    lines.append(f"| Successful | {successful} |")
    lines.append(f"| Failed | {total - successful - errors} |")
    lines.append(f"| Errors | {errors} |")
    lines.append(f"| Timed out | {timed_out} |")
    lines.append(f"| **Success rate** | **{rate:.1f}%** |")
    lines.append(f"| Total cost | ${total_cost:.4f} |")
    lines.append(f"| Total tokens | {total_tokens} |")
    lines.append("")

    # Per-repo breakdown
    by_repo: Dict[str, list] = defaultdict(list)
    for r in results:
        by_repo[r.get("repo", "unknown")].append(r)

    lines.append("## Per-Repository Breakdown")
    lines.append("")
    lines.append("| Repository | Tasks | Successful | Rate | Cost |")
    lines.append("|------------|-------|------------|------|------|")
    for repo in sorted(by_repo):
        repo_results = by_repo[repo]
        n = len(repo_results)
        s = sum(1 for r in repo_results if r.get("success"))
        c = sum(
            float(r.get("agent_result", {}).get("summary", {}).get("total_cost_usd", 0))
            for r in repo_results
        )
        lines.append(f"| {repo} | {n} | {s} | {s / n * 100:.1f}% | ${c:.4f} |")
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| Instance ID | Status | Cost | Tokens | Time | Error |")
    lines.append("|-------------|--------|------|--------|------|-------|")
    for r in results:
        iid = r.get("instance_id", "?")
        status = "✅ PASS" if r.get("success") else "❌ FAIL"
        agent = r.get("agent_result", {})
        cost = (
            f"${agent.get('summary', {}).get('total_cost_usd', '?'):.4f}"
            if isinstance(agent.get("summary", {}).get("total_cost_usd"), (int, float))
            else "?"
        )
        tok = agent.get("summary", {}).get("total_tokens", "?")
        elapsed = agent.get("elapsed_seconds", "?")
        if isinstance(elapsed, float):
            elapsed = f"{elapsed:.0f}s"
        error = r.get("error", "")
        if agent.get("timed_out"):
            error = "TIMEOUT"
        lines.append(
            f"| {iid} | {status} | {cost} | {tok} | {elapsed} | {error[:60]} |"
        )
    lines.append("")

    # Comparison
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Baseline | Score | Notes |")
    lines.append("|----------|-------|-------|")
    lines.append(
        f"| **auto-swe-agent** | **{rate:.1f}%** | {successful}/{total} on SWE-bench Lite |"
    )
    lines.append(
        "| Devin (early) | 13.86% | From [Cognition blog](https://www.cognition.ai/blog/introducing-devin) |"
    )
    lines.append(
        "| SWE-agent (default) | 12.47% | From [SWE-agent paper](https://arxiv.org/abs/2405.15793) |"
    )
    lines.append("")

    lines.append("---")
    lines.append(f"_Report generated from {len(results)} result(s)_")
    lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SWE-bench evaluation report")
    parser.add_argument("--results", nargs="*", help="Result JSON files to include")
    parser.add_argument("--output", "-o", default=None, help="Write report to file")
    args = parser.parse_args()

    if args.results:
        results: List[Dict] = []
        for p in args.results:
            data = json.loads(Path(p).read_text())
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    else:
        results = load_results()

    report = generate_report(results)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
