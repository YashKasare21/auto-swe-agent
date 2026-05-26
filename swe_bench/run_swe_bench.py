#!/usr/bin/env python3
"""CLI entry point for running auto-swe-agent on SWE-bench Lite.

Usage:
    python -m swe_bench.run_swe_bench --num-tasks 10
    python -m swe_bench.run_swe_bench --instance-ids django__django-11011 django__django-11039
    python -m swe_bench.run_swe_bench --num-tasks 5 --budget 3.0 --single-agent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_bench.harness import SWEBenchHarness


def main():
    parser = argparse.ArgumentParser(
        description="Run auto-swe-agent on SWE-bench Lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m swe_bench.run_swe_bench --num-tasks 10\n"
            "  python -m swe_bench.run_swe_bench --instance-ids django__django-11011\n"
            "  python -m swe_bench.run_swe_bench --num-tasks 5 --budget 3.0 --single-agent\n"
        ),
    )
    parser.add_argument(
        "--num-tasks", type=int, default=10,
        help="Number of tasks to evaluate (default: 10, use -1 for all)",
    )
    parser.add_argument(
        "--instance-ids", nargs="+",
        help="Specific instance IDs to evaluate (overrides --num-tasks)",
    )
    parser.add_argument(
        "--budget", type=float, default=5.0,
        help="Dollar budget per agent run (default: 5.0)",
    )
    parser.add_argument(
        "--single-agent", action="store_true",
        help="Use single-agent mode instead of multi-agent",
    )
    parser.add_argument(
        "--agent-timeout", type=int, default=1800,
        help="Agent timeout in seconds per task (default: 1800)",
    )
    parser.add_argument(
        "--results-dir", default="swe_bench/results",
        help="Directory to save results (default: swe_bench/results)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write summary JSON (default: --results-dir/summary.json)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Do not remove temporary workspaces after evaluation",
    )
    args = parser.parse_args()

    # Convert -1 (all tasks) to None
    num_tasks = args.num_tasks if args.num_tasks and args.num_tasks != -1 else None

    harness = SWEBenchHarness(
        results_dir=args.results_dir,
        agent_timeout=args.agent_timeout,
    )
    summary = harness.run_evaluation(
        num_tasks=num_tasks,
        instance_ids=args.instance_ids,
        budget=args.budget,
        single_agent=args.single_agent,
        cleanup=not args.no_cleanup,
    )

    # Write summary
    output_path = args.output or str(Path(args.results_dir) / "summary.json")
    Path(output_path).write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {output_path}")

    # Print report
    print_report(summary)


def print_report(summary: dict) -> None:
    print(f"\n{'='*60}")
    print("SWE-bench Lite Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total tasks:     {summary['total_tasks']}")
    print(f"Successful:      {summary['successful']}")
    print(f"Failed:          {summary['failed']}")
    print(f"Errors:          {summary['errors']}")
    print(f"Timed out:       {summary['timed_out']}")
    print(f"Success rate:    {summary['success_rate']:.1%}")
    print(f"Total cost:      ${summary['total_cost_usd']:.4f}")
    print(f"Total tokens:    {summary['total_tokens']}")
    print()

    if summary.get("results"):
        print("Per-task results:")
        print(f"  {'Instance ID':<40} {'Status':<6} {'Cost':>8} {'Tok':>8} {'Time':>7}")
        print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")
        for r in summary["results"]:
            iid = r.get("instance_id", "?")
            status = "PASS" if r.get("success") else "FAIL"
            agent = r.get("agent_result", {})
            cost = agent.get("summary", {}).get("total_cost_usd", "?")
            tok = agent.get("summary", {}).get("total_tokens", "?")
            elapsed = agent.get("elapsed_seconds", "?")
            if isinstance(elapsed, float):
                elapsed = f"{elapsed:.0f}s"
            extra = ""
            if r.get("error"):
                extra = f" err={r['error'][:40]}"
            elif isinstance(cost, str):
                extra = " no-cost"
            print(f"  {iid:<40} {status:<6} {str(cost):>8} {str(tok):>8} {str(elapsed):>7}{extra}")


if __name__ == "__main__":
    main()
