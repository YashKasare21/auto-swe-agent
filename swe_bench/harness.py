"""SWE-bench Lite evaluation harness for auto-swe-agent.

Loads tasks from the SWE-bench Lite dataset, sets up workspaces by cloning
repos at their base commits, runs the agent, extracts patches, and evaluates
against the dataset's fail-to-pass tests.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

from observability.langfuse_client import get_langfuse


class SWEBenchHarness:
    """Harness for evaluating auto-swe-agent on SWE-bench Lite tasks.

    Usage:
        harness = SWEBenchHarness()
        summary = harness.run_evaluation(num_tasks=10)
    """

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        results_dir: str = "swe_bench/results",
        agent_timeout: int = 1800,
        clone_depth: int = 1,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.agent_timeout = agent_timeout
        self.clone_depth = clone_depth
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            from datasets import load_dataset
            print(f"[SWE-bench] Loading {self.dataset_name} ({self.split} split)...")
            self._dataset = list(load_dataset(self.dataset_name, split=self.split))
            print(f"[SWE-bench] Loaded {len(self._dataset)} tasks")
        return self._dataset

    def get_task(self, instance_id: str) -> Dict[str, Any]:
        """Get a single task by its instance_id (e.g. 'django__django-11011')."""
        for item in self.dataset:
            if item.get("instance_id") == instance_id:
                return item
        raise ValueError(
            f"Instance {instance_id} not found in {self.dataset_name} ({self.split} split)"
        )

    def get_tasks(self, num_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the first N tasks (or all if num_tasks is None)."""
        tasks = list(self.dataset)
        if num_tasks is not None:
            return tasks[:num_tasks]
        return tasks

    def get_tasks_by_ids(self, instance_ids: List[str]) -> List[Dict[str, Any]]:
        """Return tasks matching the given instance_ids."""
        ids_set = set(instance_ids)
        return [t for t in self.dataset if t.get("instance_id") in ids_set]

    # ------------------------------------------------------------------
    # Workspace setup
    # ------------------------------------------------------------------

    def _parse_repo(self, repo: str) -> tuple[str, str]:
        """Parse 'django/django' into ('django', 'django')."""
        parts = repo.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], repo.replace("/", "-")

    def setup_workspace(self, task: Dict[str, Any]) -> Path:
        """Clone the repo at the pre-bug commit into a temp directory.

        Returns the Path to the workspace.
        """
        instance_id = task["instance_id"]
        repo = task["repo"]
        base_commit = task.get("base_commit", task.get("base_commit", "HEAD"))

        org, repo_name = self._parse_repo(repo)
        workspace = Path(tempfile.mkdtemp(prefix=f"swe-{instance_id}-"))

        clone_url = f"https://github.com/{repo}.git"
        print(f"  [SETUP] Cloning {clone_url} @ {base_commit[:12]}...")

        result = subprocess.run(
            ["git", "clone", clone_url, str(workspace),
             "--depth", str(self.clone_depth)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            shutil.rmtree(workspace, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone {clone_url}: {result.stderr.strip()[:200]}"
            )

        checkout = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=workspace, capture_output=True, text=True, timeout=120,
        )
        if checkout.returncode != 0:
            # Try fetching the commit first
            subprocess.run(
                ["git", "fetch", "--depth", "50", "origin", base_commit],
                cwd=workspace, capture_output=True, text=True, timeout=120,
            )
            checkout = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=workspace, capture_output=True, text=True, timeout=120,
            )
            if checkout.returncode != 0:
                shutil.rmtree(workspace, ignore_errors=True)
                raise RuntimeError(
                    f"Failed to checkout {base_commit}: "
                    f"{checkout.stderr.strip()[:200]}"
                )

        print(f"  [SETUP] Workspace ready at {workspace}")
        print(f"  [SETUP] Repo head: "
              f"{subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=workspace, capture_output=True, text=True).stdout.strip()}")
        return workspace

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def run_agent_on_task(
        self,
        task: Dict[str, Any],
        workspace: Path,
        budget: float = 5.0,
        single_agent: bool = False,
    ) -> Dict[str, Any]:
        """Run auto-swe-agent on a single SWE-bench task.

        Returns a dict with agent stdout, stderr, returncode, and cost info.
        """
        issue = task["problem_statement"]
        agent_py = Path(__file__).parent.parent / "agent.py"

        cmd = [
            sys.executable, str(agent_py),
            issue,
            "--workspace", str(workspace),
            "--budget", str(budget),
        ]
        if single_agent:
            cmd.append("--single-agent")
        env = {**os.environ}

        print(f"  [AGENT] Running (timeout={self.agent_timeout}s, budget=${budget})...")
        start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.agent_timeout, cwd=str(Path(__file__).parent.parent),
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            print(f"  [AGENT] TIMEOUT after {self.agent_timeout}s")
            return {
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + "\n[TIMEOUT]",
                "returncode": -1,
                "timed_out": True,
            }
        elapsed = time.time() - start
        print(f"  [AGENT] Finished in {elapsed:.1f}s (exit {result.returncode})")

        # Parse summary line for cost/token info
        summary = {}
        for line in (result.stdout or "").splitlines():
            if "[SUMMARY]" in line:
                for part in line.split("|"):
                    part = part.strip()
                    if "=" in part:
                        k, v = part.split("=", 1)
                        summary[k.strip().lower()] = v.strip()
                break

        return {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "returncode": result.returncode,
            "timed_out": False,
            "elapsed_seconds": elapsed,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def extract_patch(self, workspace: Path) -> str:
        """Return `git diff HEAD` from the workspace."""
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=workspace, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  [PATCH] git diff failed: {result.stderr.strip()[:200]}")
            return ""
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=workspace, capture_output=True, text=True, timeout=30,
        )
        new_files = ""
        if untracked.stdout.strip():
            for f in untracked.stdout.strip().splitlines():
                fpath = workspace / f
                if fpath.is_file():
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        new_files += f"--- /dev/null\n+++ b/{f}\n@@ -0,0 +1,{len(content.splitlines())} @@\n"
                        for line in content.splitlines():
                            new_files += f"+{line}\n"
                    except Exception:
                        pass
        patch = result.stdout + new_files
        print(f"  [PATCH] Extracted {len(patch)} bytes of diff")
        return patch

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_patch(
        self,
        task: Dict[str, Any],
        patch: str,
        workspace: Path,
    ) -> Dict[str, Any]:
        """Evaluate a patch by applying it and running fail-to-pass tests.

        This is a simplified evaluator. For production, use SWE-bench's
        official Docker-based evaluation harness.
        """
        eval_report = {
            "patch_applied": False,
            "tests_passed": None,
            "test_output": "",
            "setup_errors": [],
            "test_cmd": "",
        }

        if not patch.strip():
            print("  [EVAL] No patch to evaluate")
            eval_report["setup_errors"].append("Empty patch")
            return eval_report

        # Apply patch
        apply_result = subprocess.run(
            ["git", "apply", "--check"],
            cwd=workspace, input=patch, text=True,
            capture_output=True, timeout=30,
        )
        if apply_result.returncode != 0:
            print(f"  [EVAL] Patch does not apply cleanly: "
                  f"{apply_result.stderr.strip()[:150]}")
            eval_report["setup_errors"].append(
                f"Patch reject: {apply_result.stderr.strip()[:150]}"
            )
            return eval_report

        subprocess.run(
            ["git", "apply"],
            cwd=workspace, input=patch, text=True,
            capture_output=True, timeout=30,
        )
        eval_report["patch_applied"] = True
        print("  [EVAL] Patch applied cleanly")

        # Determine test command from dataset
        test_cmd = task.get("test_cmd", "").strip()
        if not test_cmd:
            test_cmd = "pytest -x -q 2>&1" 
        eval_report["test_cmd"] = test_cmd

        # Run tests
        print(f"  [EVAL] Running: {test_cmd}")
        test_result = subprocess.run(
            ["bash", "-c", test_cmd],
            cwd=workspace, capture_output=True, text=True, timeout=300,
        )
        test_output = (test_result.stdout or "") + (test_result.stderr or "")
        tests_passed = test_result.returncode == 0
        eval_report["tests_passed"] = tests_passed
        eval_report["test_output"] = test_output[:2000]

        if tests_passed:
            print("  [EVAL] Tests PASSED")
        else:
            err_lines = test_output.strip().splitlines()[-5:]
            print(f"  [EVAL] Tests FAILED (exit {test_result.returncode})")
            for l in err_lines:
                print(f"    {l}")

        return eval_report

    # ------------------------------------------------------------------
    # Full evaluation loop
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        num_tasks: Optional[int] = None,
        instance_ids: Optional[List[str]] = None,
        budget: float = 5.0,
        single_agent: bool = False,
        max_workers: int = 1,
        cleanup: bool = True,
    ) -> Dict[str, Any]:
        """Run the full SWE-bench evaluation.

        Args:
            num_tasks: Number of tasks to evaluate (first N from dataset).
            instance_ids: Specific instance IDs to evaluate (overrides num_tasks).
            budget: Dollar budget per agent run.
            single_agent: Use single-agent mode instead of multi-agent.
            max_workers: Parallel workers (1=sequential, default).
            cleanup: Remove temp workspaces after evaluation.

        Returns:
            Summary dict with total_tasks, successful, failed, errors, results.
        """
        if instance_ids:
            tasks = self.get_tasks_by_ids(instance_ids)
        else:
            tasks = self.get_tasks(num_tasks)

        print(f"\n{'='*60}")
        print(f"SWE-bench Lite Evaluation — {len(tasks)} tasks")
        print(f"{'='*60}\n")

        all_results: List[Dict[str, Any]] = []
        langfuse = get_langfuse()

        for idx, task in enumerate(tasks, 1):
            instance_id = task.get("instance_id", "unknown")
            repo = task.get("repo", "unknown")

            # Create Langfuse trace for this task
            eval_trace = langfuse.create_trace(
                name=f"swe-bench-{instance_id}",
                metadata={
                    "instance_id": instance_id,
                    "repo": repo,
                    "task_idx": idx,
                    "num_tasks": len(tasks),
                },
            )
            eval_trace_id = eval_trace.id if eval_trace is not None else None

            print(f"[{idx}/{len(tasks)}] {instance_id} ({repo})")

            result: Dict[str, Any] = {
                "instance_id": instance_id,
                "repo": repo,
                "index": idx,
            }
            workspace: Optional[Path] = None

            try:
                workspace = self.setup_workspace(task)
                agent_result = self.run_agent_on_task(
                    task, workspace, budget=budget, single_agent=single_agent,
                )
                patch = self.extract_patch(workspace)

                if agent_result.get("timed_out"):
                    eval_report = {
                        "patch_applied": False,
                        "tests_passed": False,
                        "test_output": "",
                        "setup_errors": ["Agent timed out"],
                        "test_cmd": "",
                    }
                else:
                    eval_report = self.evaluate_patch(task, patch, workspace)

                result["patch"] = patch
                result["agent_result"] = agent_result
                result["evaluation"] = eval_report
                result["success"] = (
                    eval_report["tests_passed"] is True
                )

            except Exception as e:
                print(f"  [ERROR] {type(e).__name__}: {e}")
                result["error"] = str(e)
                result["success"] = False

            # Score in Langfuse
            if eval_trace_id:
                langfuse.score(
                    trace_id=eval_trace_id,
                    name="tests_passed",
                    value=1.0 if result.get("success") else 0.0,
                    comment=(
                        f"repo={repo} | "
                        f"error={str(result.get('error', ''))[:100] if result.get('error') else 'none'}"
                    ),
                )

            all_results.append(result)

            # Save incremental results
            self._save_results(all_results)

            # Print per-task summary
            status = "PASS" if result.get("success") else "FAIL"
            extra = ""
            if result.get("error"):
                extra = f" ({result['error'][:60]})"
            elif "agent_result" in result:
                agent_r = result["agent_result"]
                if agent_r.get("summary"):
                    s = agent_r["summary"]
                    extra = f" cost=${s.get('total_cost_usd', '?')} tok={s.get('total_tokens', '?')}"
                if agent_r.get("timed_out"):
                    extra += " [TIMEOUT]"
            print(f"  => {status}{extra}")
            print()

            if cleanup and workspace is not None and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

        langfuse.flush()
        return self._summarize_results(all_results)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_results(self, results: List[Dict[str, Any]]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"swe_bench_results_{timestamp}.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\n  [SAVE] Incremental results -> {path}")
        return path

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        successful = sum(1 for r in results if r.get("success"))
        errors = sum(1 for r in results if "error" in r)
        timed_out = sum(
            1 for r in results
            if r.get("agent_result", {}).get("timed_out")
        )
        total_cost = 0.0
        total_tokens = 0
        for r in results:
            agent = r.get("agent_result", {})
            if isinstance(agent, dict):
                summary = agent.get("summary", {})
                try:
                    total_cost += float(summary.get("total_cost_usd", 0))
                except (ValueError, TypeError):
                    pass
                try:
                    total_tokens += int(summary.get("total_tokens", 0))
                except (ValueError, TypeError):
                    pass

        return {
            "total_tasks": total,
            "successful": successful,
            "failed": total - successful - errors,
            "errors": errors,
            "timed_out": timed_out,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "results": results,
        }



